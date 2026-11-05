"""Solving many instances in parallel, and comparing every solution with the closed form."""

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import yaml

from ace_sddp.model import Economy, Instance
from ace_sddp.sddp import SDDP
from ace_sddp.stage import StageProblem

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
_WORKER = {}


def load_config(version: str) -> dict:
    with open(CONFIGS / f"ace_dice_{version}.yaml") as f:
        return yaml.safe_load(f)


def compare(inst: Instance, path: dict, closed: dict) -> dict:
    """Errors of a solution path against the closed form, over the evaluated periods."""
    n = inst.economy.evaluate
    scc_err = np.abs(path["scc"][:n] / closed["scc"][:n] - 1)
    # Consumption-equivalent welfare loss over the full horizon
    loss = float(np.expm1((closed["value"] - path["value"]) * (1 - inst.beta)))
    return {
        "x_max_abs_err": float(np.max(np.abs(path["x"][:n] - closed["x"][:n]))),
        "mu_mean_abs_err": float(np.mean(np.abs(path["mu"][:n] - closed["mu"][:n]))),
        "mu_max_abs_err": float(np.max(np.abs(path["mu"][:n] - closed["mu"][:n]))),
        "scc_mean_rel_err": float(np.mean(scc_err)),
        "scc_initial_rel_err": float(scc_err[0]),
        "scc_initial": float(path["scc"][0]),
        "scc_initial_star": float(closed["scc"][0]),
        "welfare_loss": loss,
    }


def _init(version: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = load_config(version)
    _WORKER.update(cfg=cfg, economy=Economy(cfg), stage=StageProblem(cfg["sddp"]["max_cuts"], cfg["sddp"]["consumption_bounds"]))


def _cold(parameters: np.ndarray) -> dict:
    """SDDP from scratch; returns the errors, the cost and the final cut of every stage."""
    cfg, eco, stage = _WORKER["cfg"], _WORKER["economy"], _WORKER["stage"]
    inst = Instance(eco, parameters)
    result = SDDP(inst, stage, cfg).run()
    out = compare(inst, result.path, inst.closed_form_path())
    out.update(iterations=result.iterations, solves=result.solves, seconds=result.seconds,
               first_lower=result.history[0]["lower"], lower=result.lower, upper=result.upper)
    out["cuts"] = np.array([c[-1] for c in result.cuts[: eco.horizon]])
    return out


def _neural(args) -> dict:
    """The predicted cuts alone (one forward pass), then SDDP warm-started from them."""
    parameters, cuts = args
    cfg, eco, stage = _WORKER["cfg"], _WORKER["economy"], _WORKER["stage"]
    inst = Instance(eco, parameters)
    closed = inst.closed_form_path()

    sddp = SDDP(inst, stage, cfg)
    sddp.warm_start(cuts)
    start = time.time()
    path = sddp.forward()
    zero_shot = compare(inst, path, closed)
    zero_shot.update(seconds=time.time() - start, solves=sddp.solves, value=path["value"])

    warm = SDDP(inst, stage, cfg)
    warm.warm_start(cuts)
    result = warm.run()
    warm_out = compare(inst, result.path, closed)
    warm_out.update(iterations=result.iterations, solves=result.solves, seconds=result.seconds,
                    first_lower=result.history[0]["lower"], lower=result.lower, upper=result.upper)
    return {"zero_shot": zero_shot, "warm": warm_out}


def run_parallel(version: str, function, items, workers: int = None) -> list:
    workers = workers or os.cpu_count() or 1
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(version,)) as pool:
        return pool.map(function, items, chunksize=1)


def solve_cold(version: str, draws: np.ndarray) -> list:
    return run_parallel(version, _cold, list(draws))


def solve_neural(version: str, draws: np.ndarray, cuts: list) -> list:
    return run_parallel(version, _neural, list(zip(draws, cuts)))
