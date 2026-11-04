"""Run every experiment of part 6 and save the raw results.

Usage: python run.py [--version 2016 2023]

For each version of the model:
  1. SDDP from scratch on 128 training draws of the parameters; their cuts are the training data.
  2. Train the cut network on those cuts.
  3. On 256 test draws (the same Sobol points as part 5): SDDP from scratch, the network's
     cuts alone (no SDDP iterations), and SDDP warm-started from the network's cuts.
  4. Convergence of SDDP at the baseline, from scratch and warm-started.
  5. SDDP with productivity shocks at the baseline.

Writes ../results/runs/dice<version>/; make_results.py turns them into figures and tables.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ace_sddp.experiments import load_config, solve_cold, solve_neural
from ace_sddp.model import PARAMETERS, Economy, Instance
from ace_sddp.sddp import SDDP
from ace_sddp.stage import StageProblem

RESULTS = Path(__file__).resolve().parent.parent / "results" / "runs"
TEST_DRAWS = 256
TEST_SEED = 7  # Same test points as part 5
TRAIN_SEED = 11
SHOCK_SD = 0.03


def draws(economy: Economy, n: int, seed: int) -> np.ndarray:
    return economy.low + (economy.high - economy.low) * qmc.Sobol(3, seed=seed).random(n)


def table(results: list, parameters: np.ndarray) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if k != "cuts"} for r in results]
    frame = pd.DataFrame(rows)
    for i, name in enumerate(PARAMETERS):
        frame.insert(i, name, parameters[:, i])
    return frame


def run(version: str) -> None:
    from ace_sddp.neural import CutNetwork

    cfg = load_config(version)
    eco = Economy(cfg)
    out = RESULTS / f"dice{version}"
    out.mkdir(parents=True, exist_ok=True)
    T = eco.horizon
    timing = {}

    # 1. Training data from SDDP
    train = draws(eco, cfg["neural"]["train_draws"], TRAIN_SEED)
    start = time.time()
    train_results = solve_cold(version, train)
    timing["train_sddp_seconds"] = time.time() - start
    table(train_results, train).to_csv(out / "train_sddp.csv", index=False)
    cuts = np.stack([r["cuts"] for r in train_results])  # [draws, T, 7]
    np.savez_compressed(out / "train_cuts.npz", parameters=train, cuts=cuts)
    print(f"DICE {version}: training SDDP done in {timing['train_sddp_seconds']:.0f} s", flush=True)

    # 2. The cut network
    network = CutNetwork(cfg, eco.low, eco.high, T)
    stages = np.tile(np.arange(T), len(train))
    start = time.time()
    history = network.fit(np.repeat(train, T, axis=0), stages, cuts.reshape(-1, 7))
    timing["train_network_seconds"] = time.time() - start
    network.save(out / "network")
    (out / "network_history.json").write_text(json.dumps(history))
    print(f"DICE {version}: network trained in {timing['train_network_seconds']:.0f} s, loss {history['loss'][-1]:.2e}", flush=True)

    # 3. Test draws
    test = draws(eco, TEST_DRAWS, TEST_SEED)
    start = time.time()
    cold = solve_cold(version, test)
    timing["test_sddp_seconds"] = time.time() - start
    table(cold, test).to_csv(out / "test_sddp.csv", index=False)

    start = time.time()
    predicted = [network.cuts_for(p) for p in test]
    timing["predict_seconds_per_draw"] = (time.time() - start) / len(test)
    neural = solve_neural(version, test, predicted)
    table([r["zero_shot"] for r in neural], test).to_csv(out / "test_neural.csv", index=False)
    table([r["warm"] for r in neural], test).to_csv(out / "test_warm.csv", index=False)
    print(f"DICE {version}: test draws done", flush=True)

    # 4. Convergence at the baseline
    stage = StageProblem(cfg["sddp"]["max_cuts"], cfg["sddp"]["consumption_bounds"])
    inst = Instance(eco, eco.baseline)
    cold_history = SDDP(inst, stage, cfg).run(max_iterations=4, tolerance=0.0).history
    warm_sddp = SDDP(inst, stage, cfg)
    warm_sddp.warm_start(network.cuts_for(eco.baseline))
    warm_history = warm_sddp.run(max_iterations=4, tolerance=0.0).history
    closed = inst.closed_form_path()
    (out / "convergence.json").write_text(json.dumps({"cold": cold_history, "warm": warm_history, "optimum": closed["value"]}))

    # 5. Productivity shocks: the slopes and the policy should not change
    shocked = SDDP(inst, stage, cfg, shock_sd=SHOCK_SD, seed=1).run(max_iterations=6)
    cf = inst.closed_form()
    n = eco.evaluate
    stochastic = {
        "shock_sd": SHOCK_SD,
        "iterations": shocked.iterations,
        "solves": shocked.solves,
        "seconds": shocked.seconds,
        "slope_rel_err": float(np.max(np.abs(shocked.path["slopes_next"][:n] / cf.slopes - 1))),
        "x_max_abs_err": float(np.max(np.abs(shocked.path["x"][:n] - cf.x))),
        "mu_max_abs_err": float(np.max(np.abs(shocked.path["mu"][:n] - closed["mu"][:n]))),
        "k_path_range": [float(np.min(shocked.path["states"][:n, 0] - closed["states"][:n, 0])),
                         float(np.max(shocked.path["states"][:n, 0] - closed["states"][:n, 0]))],
    }
    (out / "stochastic.json").write_text(json.dumps(stochastic))
    (out / "timing.json").write_text(json.dumps(timing))
    print(f"DICE {version}: done", json.dumps(timing), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", nargs="+", default=["2016", "2023"])
    args = parser.parse_args()
    for version in args.version:
        run(version)


if __name__ == "__main__":
    main()
