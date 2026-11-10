"""How the horizon truncation affects the SCC: SDDP with horizons of 100 to 450 periods.

Usage: python truncation.py
Writes ../results/tables/truncation.csv: the SCC error against the closed form (infinite
horizon) over the periods up to 2300, for three parameter draws of DICE 2016.
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from ace_sddp.experiments import load_config
from ace_sddp.model import Economy, Instance
from ace_sddp.sddp import SDDP
from ace_sddp.stage import StageProblem

RESULTS = Path(__file__).resolve().parent.parent / "results" / "tables"
HORIZONS = [100, 150, 250, 350, 450]
CASES = {
    "rho 0.5%, xi_0 x 2, exchange 0.75": [0.005, 0.044, 0.75],
    "baseline": None,
    "rho 3.5%, xi_0 x 0.5, exchange 1.25": [0.035, 0.011, 1.25],
}


def main() -> None:
    base = load_config("2016")
    rows = []
    for horizon in HORIZONS:
        cfg = copy.deepcopy(base)
        cfg["simulation"]["horizon"] = horizon
        eco = Economy(cfg)
        stage = StageProblem(cfg["sddp"]["max_cuts"], cfg["sddp"]["consumption_bounds"])
        for case, parameters in CASES.items():
            inst = Instance(eco, eco.baseline if parameters is None else parameters)
            path = SDDP(inst, stage, cfg).run().path
            n = eco.evaluate
            err = np.abs(path["scc"][:n] / inst.closed_form_path(n)["scc"] - 1)
            rows.append({"horizon": horizon, "case": case, "scc_mean_rel_err": err.mean(), "scc_max_rel_err": err.max()})
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "truncation.csv", index=False)
    print(frame.to_string())


if __name__ == "__main__":
    main()
