"""Figures and tables of part 6 from the raw results of run.py.

Usage: python make_results.py
Reads   ../results/runs/dice{2016,2023}/ and, for comparison, ../../5_deep_uq/results/csv/
Writes  ../results/figures/*.pdf|png and ../results/tables/*.csv
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ace_sddp.experiments import load_config
from ace_sddp.model import Economy, Instance
from ace_sddp.neural import CutNetwork
from ace_sddp.sddp import SDDP
from ace_sddp.stage import StageProblem

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RUNS = RESULTS / "runs"
DEQN = HERE.parent.parent / "5_deep_uq" / "results" / "csv"
VERSIONS = ["2016", "2023"]
START_YEAR = {"2016": 2015, "2023": 2020}

COLOR_NEURAL = "#2a78d6"
COLOR_DEQN = "#eb6834"
COLOR_SDDP = "#1baf7a"
COLOR_ANALYTIC = "#3d3d3a"
INK_SECONDARY = "#52514e"
ANALYTIC_STYLE = dict(color=COLOR_ANALYTIC, linestyle=(0, (4, 3)), linewidth=1.3)
METRICS = ["x_max_abs_err", "mu_mean_abs_err", "mu_max_abs_err", "scc_mean_rel_err", "scc_initial_rel_err", "welfare_loss"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 10.5, "axes.edgecolor": INK_SECONDARY,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"{name}.pdf")
    fig.savefig(RESULTS / "figures" / f"{name}.png")
    plt.close(fig)


def load(version: str) -> dict:
    run = RUNS / f"dice{version}"
    data = {name: pd.read_csv(run / f"{name}.csv") for name in ("train_sddp", "test_sddp", "test_neural", "test_warm")}
    for name in ("convergence", "stochastic", "timing", "network_history"):
        data[name] = json.loads((run / f"{name}.json").read_text())
    deqn = pd.read_csv(DEQN / f"test_draws_dice{version}.csv")
    same = np.allclose(deqn[["prtp", "xi_0", "carbon_exchange"]].values, data["test_sddp"][["prtp", "xi_0", "carbon_exchange"]].values)
    data["deqn"] = deqn if same else None
    return data


################ TABLES ################
def accuracy_table(all_data: dict) -> pd.DataFrame:
    rows = []
    for version, d in all_data.items():
        methods = [("SDDP", d["test_sddp"]), ("neural SDDP, no iterations", d["test_neural"]), ("neural SDDP, warm-started", d["test_warm"])]
        if d["deqn"] is not None:
            methods.append(("pseudo-state DEQN (part 5)", d["deqn"]))
        for method, frame in methods:
            row = {"version": version, "method": method}
            for metric in METRICS:
                if metric in frame:
                    row[f"{metric}_median"] = frame[metric].median()
                    row[f"{metric}_p90"] = frame[metric].quantile(0.9)
                    row[f"{metric}_max"] = frame[metric].max()
            rows.append(row)
    return pd.DataFrame(rows)


def cost_table(all_data: dict) -> pd.DataFrame:
    rows = []
    for version, d in all_data.items():
        timing = d["timing"]
        rows.append({
            "version": version,
            "sddp_seconds_per_draw": d["test_sddp"]["seconds"].median(),
            "sddp_solves_per_draw": d["test_sddp"]["solves"].median(),
            "sddp_iterations": d["test_sddp"]["iterations"].median(),
            "neural_seconds_per_draw": d["test_neural"]["seconds"].median() + timing["predict_seconds_per_draw"],
            "neural_solves_per_draw": d["test_neural"]["solves"].median(),
            "warm_seconds_per_draw": d["test_warm"]["seconds"].median(),
            "warm_iterations": d["test_warm"]["iterations"].median(),
            "training_data_seconds": timing["train_sddp_seconds"],
            "training_network_seconds": timing["train_network_seconds"],
            "training_draws": len(d["train_sddp"]),
            "stochastic_seconds": d["stochastic"]["seconds"],
            "stochastic_slope_rel_err": d["stochastic"]["slope_rel_err"],
            "stochastic_mu_max_abs_err": d["stochastic"]["mu_max_abs_err"],
        })
    return pd.DataFrame(rows)


################ FIGURES ################
def figure_accuracy(all_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, (version, d) in zip(axes, all_data.items()):
        rho = 100 * d["test_sddp"]["prtp"]
        ax.scatter(rho, 100 * d["test_sddp"]["scc_mean_rel_err"], s=8, color=COLOR_SDDP, label="SDDP")
        ax.scatter(rho, 100 * d["test_neural"]["scc_mean_rel_err"], s=8, color=COLOR_NEURAL, label="neural SDDP, no iterations")
        if d["deqn"] is not None:
            ax.scatter(rho, 100 * d["deqn"]["scc_mean_rel_err"], s=8, color=COLOR_DEQN, alpha=0.7, label="pseudo-state DEQN (part 5)")
        ax.set_yscale("log")
        ax.set_title(f"ACE-DICE {version}: SCC error to 2300")
        ax.set_xlabel("Pure rate of time preference ρ (% per year)")
    axes[0].set_ylabel("Mean relative SCC error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), markerscale=2)
    save(fig, "accuracy")


def figure_convergence(all_data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    for ax, (version, d) in zip(axes, all_data.items()):
        conv = d["convergence"]
        optimum = conv["optimum"]
        for key, color, label in (("cold", COLOR_SDDP, "SDDP from scratch"), ("warm", COLOR_NEURAL, "warm-started by the network")):
            history = conv[key]
            shortfall = [max((optimum - h["lower"]) / abs(optimum), 1e-12) for h in history]
            ax.plot([h["solves"] for h in history], shortfall, marker="o", color=color, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Stage problems solved")
        ax.set_title(f"ACE-DICE {version}, baseline")
    axes[0].set_ylabel("Policy value below the optimum (relative)")
    axes[0].legend(loc="upper right")
    save(fig, "convergence")


def figure_parity(all_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    for color, (version, d) in zip((COLOR_NEURAL, COLOR_DEQN), all_data.items()):
        n = d["test_neural"]
        ax.scatter(n["scc_initial_star"], n["scc_initial"], s=9, color=color, alpha=0.75, label=f"DICE {version}")
    lim = [min(d["test_neural"]["scc_initial_star"].min() for d in all_data.values()) * 0.8,
           max(d["test_neural"]["scc_initial_star"].max() for d in all_data.values()) * 1.2]
    ax.plot(lim, lim, **ANALYTIC_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Closed-form SCC, first period (USD/tCO₂)")
    ax.set_ylabel("Neural SDDP, no iterations")
    ax.set_title("SCC from the predicted cuts, 256 test draws")
    ax.legend(loc="upper left")
    save(fig, "parity")


def figure_paths() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), sharex="col")
    for col, version in enumerate(VERSIONS):
        cfg = load_config(version)
        eco = Economy(cfg)
        network = CutNetwork(cfg, eco.low, eco.high, eco.horizon)
        network.load(RUNS / f"dice{version}" / "network")
        stage = StageProblem(cfg["sddp"]["max_cuts"], cfg["sddp"]["consumption_bounds"])
        cases = {
            "Low SCC: ρ 3.5%, ξ₀ × 0.5": [eco.high[0], eco.low[1], 1.0],
            "Baseline": list(eco.baseline),
            "High SCC: ρ 0.5%, ξ₀ × 2": [eco.low[0], eco.high[1], 1.0],
        }
        n = eco.evaluate
        years = START_YEAR[version] + eco.timestep * np.arange(n)
        cut = years <= 2200
        for color, (label, p) in zip((COLOR_NEURAL, COLOR_SDDP, COLOR_DEQN), cases.items()):
            inst = Instance(eco, p)
            sddp = SDDP(inst, stage, cfg)
            sddp.warm_start(network.cuts_for(np.array(p)))
            path = sddp.forward()
            closed = inst.closed_form_path(n)
            axes[0, col].plot(years[cut], path["mu"][:n][cut], color=color, label=label)
            axes[0, col].plot(years[cut], closed["mu"][cut], **ANALYTIC_STYLE)
            axes[1, col].plot(years[cut], path["scc"][:n][cut], color=color)
            axes[1, col].plot(years[cut], closed["scc"][cut], **ANALYTIC_STYLE)
        axes[0, col].set_title(f"ACE-DICE {version}: neural SDDP, no iterations")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Year")
    axes[0, 0].set_ylabel("Abatement rate μ")
    axes[1, 0].set_ylabel("SCC (USD per tCO₂)")
    axes[0, 0].plot([], [], label="Closed form", **ANALYTIC_STYLE)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    save(fig, "paths")


def main() -> None:
    all_data = {v: load(v) for v in VERSIONS}
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    accuracy = accuracy_table(all_data)
    accuracy.to_csv(RESULTS / "tables" / "accuracy.csv", index=False)
    cost = cost_table(all_data)
    cost.to_csv(RESULTS / "tables" / "cost.csv", index=False)
    figure_accuracy(all_data)
    figure_convergence(all_data)
    figure_parity(all_data)
    figure_paths()
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(accuracy[["version", "method"] + [c for c in accuracy.columns if c.startswith("scc_mean") or c.startswith("mu_mean") or c.startswith("welfare")]])
        print(cost.T)
        for version, d in all_data.items():
            print(version, "convergence", [(h["iteration"], h["solves"], h["lower"], h["upper"]) for h in d["convergence"]["cold"]],
                  [(h["iteration"], h["solves"], h["lower"]) for h in d["convergence"]["warm"]], d["convergence"]["optimum"])
            print(version, "stochastic", d["stochastic"])


if __name__ == "__main__":
    main()
