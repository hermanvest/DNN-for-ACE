"""Figures and tables for the pseudo-state DEQN and the uncertainty quantification.

Usage: python make_results.py
Reads   ../results/runs/dice{2016,2023}/{policy.weights.h5,history.json}
Writes  ../results/figures/*.pdf|png, ../results/tables/*.csv|tex, ../results/csv/*.csv
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from scipy.stats import qmc

from ace_uq.analytic import analytic_actions, analytic_trajectories
from ace_uq.model import PARAMETERS, AceDiceUQ
from ace_uq.network import build_policy_network
from ace_uq.surrogate import ActiveGP, sample_belief, sobol_indices

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
VERSIONS = ["2016", "2023"]
START_YEAR = {"2016": 2015, "2023": 2020}
HORIZON = 58  # Periods up to about 2300
TEST_DRAWS = 256
UQ_DRAWS = 100_000
SOBOL_BASE = 1 << 14
GP_INITIAL, GP_TOTAL = 8, 48
PART_2_MINUTES = 24.4  # Training time of one single-instance DEQN in part 2 (2_revised/results/runs)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_ANALYTIC = "#3d3d3a"
INK_SECONDARY = "#52514e"
ANALYTIC_STYLE = dict(color=COLOR_ANALYTIC, linestyle=(0, (4, 3)), linewidth=1.3)
LABELS = {"prtp": "Time preference ρ", "xi_0": "Damages ξ₀", "carbon_exchange": "Carbon exchange"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "figures", f"{name}.pdf"))
    fig.savefig(os.path.join(RESULTS, "figures", f"{name}.png"))
    plt.close(fig)


################ MODEL ################
class Solved:
    """A trained pseudo-state DEQN with helpers to evaluate it at any parameter vector."""

    def __init__(self, version: str, tag: str = ""):
        with open(os.path.join(HERE, "configs", f"ace_dice_{version}.yaml")) as f:
            self.config = yaml.safe_load(f)
        self.version = version
        self.model = AceDiceUQ(self.config)
        self.network = build_policy_network(self.config)
        run = os.path.join(RESULTS, "runs", f"dice{version}{tag}")
        self.network.load_weights(os.path.join(run, "policy.weights.h5"))
        with open(os.path.join(run, "history.json")) as f:
            self.history = json.load(f)
        self._step = tf.function(lambda s: self.model.step(s, self.policy(s)))

    def policy(self, s):
        return self.model.decode_actions(s, self.network(self.model.preprocess(s)))

    def simulate(self, parameters, n):
        s = self.model.initial_state(parameters)
        states = []
        for _ in range(n):
            states.append(s)
            s = self._step(s)
        return tf.stack(states, axis=1)

    def scc_initial(self, parameters, chunk=20_000):
        """SCC in the first period: a function of the parameters alone."""
        out = []
        for i in range(0, len(parameters), chunk):
            s = self.model.initial_state(parameters[i : i + chunk])
            out.append(self.model.social_cost_of_carbon(s, self.policy(s)).numpy())
        return np.concatenate(out).astype(float)

    def scc_initial_analytic(self, parameters):
        s = self.model.initial_state(parameters)
        return self.model.social_cost_of_carbon(s, analytic_actions(self.model, s)).numpy().astype(float)


################ ACCURACY ################
def path_quantities(model, states, actions):
    """x, mu and SCC along a batch of paths, shape [draws, n]."""
    draws, n = states.shape[:2]
    s = tf.reshape(tf.constant(states, tf.float32), (-1, states.shape[-1]))
    a = {k: tf.constant(np.reshape(v, (draws * n,) + v.shape[2:]), tf.float32) for k, v in actions.items()}
    mu = 1.0 - a["E"] / model.E_BAU(s)
    scc = model.social_cost_of_carbon(s, a)
    return {k: np.reshape(v.numpy(), (draws, n)) for k, v in {"x": a["x"], "mu": mu, "scc": scc}.items()}


def accuracy(solved: Solved, parameters: np.ndarray):
    model = solved.model
    states = solved.simulate(parameters, HORIZON)
    flat = tf.reshape(states, (-1, states.shape[-1]))
    actions = {k: np.reshape(v.numpy(), states.shape[:2] + v.shape[1:]) for k, v in solved.policy(flat).items()}
    deqn = path_quantities(model, states.numpy(), actions)
    path = analytic_trajectories(model, parameters, n=HORIZON)
    star = path_quantities(model, path["states"], path["actions"])
    scc_err = np.abs(deqn["scc"] / star["scc"] - 1)
    per_draw = pd.DataFrame(parameters, columns=PARAMETERS)
    per_draw["x_max_abs_err"] = np.abs(deqn["x"] - star["x"]).max(axis=1)
    per_draw["mu_mean_abs_err"] = np.abs(deqn["mu"] - star["mu"]).mean(axis=1)
    per_draw["mu_max_abs_err"] = np.abs(deqn["mu"] - star["mu"]).max(axis=1)
    per_draw["scc_mean_rel_err"] = scc_err.mean(axis=1)
    per_draw["scc_initial_rel_err"] = scc_err[:, 0]
    per_draw["scc_initial"] = deqn["scc"][:, 0]
    per_draw["scc_initial_star"] = star["scc"][:, 0]
    return per_draw, deqn, star


################ FIGURES ################
def figure_slices(solved):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, version in zip(axes, VERSIONS):
        s = solved[version]
        model = s.model
        rho = np.linspace(model.low[0], model.high[0], 61)
        for color, (label, xi) in zip(
            COLORS, [("ξ₀ × 0.5", model.low[1]), ("ξ₀ baseline", model.baseline[1]), ("ξ₀ × 2", model.high[1])]
        ):
            p = np.stack([rho, np.full_like(rho, xi), np.ones_like(rho)], axis=1)
            ax.plot(100 * rho, s.scc_initial(p), color=color, label=label)
            ax.plot(100 * rho, s.scc_initial_analytic(p), **ANALYTIC_STYLE)
        ax.axvline(100 * model.baseline[0], color=INK_SECONDARY, linewidth=0.6, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(f"ACE-DICE {version}, SCC in {START_YEAR[version]}")
        ax.set_xlabel("Pure rate of time preference ρ (% per year)")
    axes[0].set_ylabel("USD per tCO₂")
    axes[0].plot([], [], label="Closed form", **ANALYTIC_STYLE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    save(fig, "scc_slices")


def figure_accuracy(accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for color, version in zip(COLORS, VERSIONS):
        d = accuracies[version]
        axes[0].scatter(d["scc_initial_star"], d["scc_initial"], s=9, color=color, alpha=0.7, label=f"DICE {version}")
        axes[1].scatter(100 * d["prtp"], 100 * d["scc_mean_rel_err"], s=9, color=color, alpha=0.7, label=f"DICE {version}")
    lim = [min(d["scc_initial_star"].min() for d in accuracies.values()) * 0.8,
           max(d["scc_initial_star"].max() for d in accuracies.values()) * 1.2]
    axes[0].plot(lim, lim, **ANALYTIC_STYLE)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Closed-form SCC (USD per tCO₂)")
    axes[0].set_ylabel("DEQN SCC (USD per tCO₂)")
    axes[0].set_title(f"SCC in the first period, {TEST_DRAWS} test draws")
    axes[1].set_xlabel("Pure rate of time preference ρ (% per year)")
    axes[1].set_ylabel("Mean relative SCC error (%)")
    axes[1].set_title("SCC error along the path to 2300")
    axes[1].legend(loc="upper right")
    save(fig, "accuracy")


def figure_paths(solved):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8), sharex="col")
    for col, version in enumerate(VERSIONS):
        s = solved[version]
        model = s.model
        cases = {
            "Low SCC: ρ 3.5%, ξ₀ × 0.5": [model.high[0], model.low[1], 1.0],
            "Baseline": list(model.baseline),
            "High SCC: ρ 0.5%, ξ₀ × 2": [model.low[0], model.high[1], 1.0],
        }
        p = np.array(list(cases.values()))
        _, deqn, star = accuracy(s, p)
        years = START_YEAR[version] + model.timestep * np.arange(HORIZON)
        cut = years <= 2200
        for i, (color, label) in enumerate(zip(COLORS, cases)):
            axes[0, col].plot(years[cut], deqn["mu"][i][cut], color=color, label=label)
            axes[0, col].plot(years[cut], star["mu"][i][cut], **ANALYTIC_STYLE)
            axes[1, col].plot(years[cut], deqn["scc"][i][cut], color=color)
            axes[1, col].plot(years[cut], star["scc"][i][cut], **ANALYTIC_STYLE)
        axes[0, col].set_title(f"ACE-DICE {version}")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Year")
    axes[0, 0].set_ylabel("Abatement rate μ")
    axes[1, 0].set_ylabel("SCC (USD per tCO₂)")
    axes[0, 0].plot([], [], label="Closed form", **ANALYTIC_STYLE)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    save(fig, "paths")


def figure_uq(uq):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.8))
    for col, version in enumerate(VERSIONS):
        u = uq[version]
        ax = axes[0, col]
        bins = np.geomspace(min(u["samples"]["analytic"].min(), u["samples"]["deqn"].min()),
                            max(u["samples"]["analytic"].max(), u["samples"]["deqn"].max()), 60)
        for color, key, label in [(COLORS[0], "deqn", "DEQN"), (COLORS[1], "gp", f"GP surrogate ({GP_TOTAL} points)")]:
            ax.hist(u["samples"][key], bins=bins, density=True, histtype="step", color=color, linewidth=1.5, label=label)
        ax.hist(u["samples"]["analytic"], bins=bins, density=True, histtype="step", label="Closed form", **ANALYTIC_STYLE)
        ax.set_xscale("log")
        ax.set_title(f"ACE-DICE {version}: SCC in {START_YEAR[version]} under the belief")
        ax.set_xlabel("USD per tCO₂")
        ax.set_yticks([])

        ax = axes[1, col]
        width = 0.26
        x = np.arange(len(PARAMETERS))
        for i, (color, key, label) in enumerate(
            [(COLORS[0], "deqn", "DEQN"), (COLORS[1], "gp", "GP surrogate"), (COLOR_ANALYTIC, "analytic", "Closed form")]
        ):
            ax.bar(x + (i - 1) * width, u["sobol"][key]["total"], width, color=color, label=label)
        ax.set_xticks(x, [LABELS[p] for p in PARAMETERS])
        ax.set_ylim(0, 1)
        ax.set_title("Total Sobol index of the SCC")
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    axes[1, 0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "uq")


def figure_learning(solved, uq):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for color, version in zip(COLORS, VERSIONS):
        h = solved[version].history
        it = np.array([v["iteration"] for v in h["validation"]])
        minutes = np.array(h["seconds"])[it] / 60
        axes[0].plot(minutes, [v["loss"] for v in h["validation"]], color=color, label=f"DICE {version}")
        trace = pd.DataFrame(uq[version]["trace"])
        axes[1].plot(trace["points"], 100 * trace["error_deqn"], color=color, label=f"DICE {version}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training time (minutes)")
    axes[0].set_title("Validation objective")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("GP design points")
    axes[1].set_title("Surrogate error vs the DEQN (max, %)")
    save(fig, "learning")


################ UNCERTAINTY QUANTIFICATION ################
def quantify(s: Solved):
    model = s.model
    belief = s.config["uncertain_parameters"]
    rng = np.random.default_rng(2024)
    check = model.low + (model.high - model.low) * qmc.Sobol(3, seed=3).random(512)
    check_deqn = s.scc_initial(check)

    def monitor(gp):
        return {"error_deqn": float(np.max(np.abs(gp.predict(check) / check_deqn - 1)))}

    gp = ActiveGP(s.scc_initial, model.low, model.high, seed=0)
    gp.fit(GP_INITIAL, GP_TOTAL, monitor=monitor)

    draws = sample_belief(belief, PARAMETERS, UQ_DRAWS, rng)
    samples = {"analytic": s.scc_initial_analytic(draws), "deqn": s.scc_initial(draws), "gp": gp.predict(draws)}

    def sampler(n):
        return sample_belief(belief, PARAMETERS, n, rng)

    sobol = {
        "analytic": sobol_indices(s.scc_initial_analytic, sampler, SOBOL_BASE),
        "deqn": sobol_indices(s.scc_initial, sampler, SOBOL_BASE),
        "gp": sobol_indices(gp.predict, sampler, SOBOL_BASE),
    }
    return {"samples": samples, "sobol": sobol, "trace": gp.trace, "gp_points": len(gp.y)}


################ TABLES ################
def write_tables(solved, accuracies, baseline, uq, ablation):
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    rows = []
    for version in VERSIONS:
        d = accuracies[version]
        b = baseline[version].iloc[0]
        for metric in ["x_max_abs_err", "mu_mean_abs_err", "mu_max_abs_err", "scc_mean_rel_err", "scc_initial_rel_err"]:
            rows.append(
                {
                    "version": version,
                    "metric": metric,
                    "baseline": b[metric],
                    "median": d[metric].median(),
                    "p90": d[metric].quantile(0.9),
                    "max": d[metric].max(),
                }
            )
        d.to_csv(os.path.join(RESULTS, "csv", f"test_draws_dice{version}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tables", "accuracy.csv"), index=False)

    rows = []
    for label, d in [("equal weights", accuracies["2016"]), ("discount weighting", ablation)]:
        low = d["prtp"] <= 0.01
        rows.append(
            {
                "weights": label,
                "scc_mean_rel_err_median": d["scc_mean_rel_err"].median(),
                "scc_mean_rel_err_p90": d["scc_mean_rel_err"].quantile(0.9),
                "scc_mean_rel_err_max": d["scc_mean_rel_err"].max(),
                "scc_mean_rel_err_median_low_prtp": d.loc[low, "scc_mean_rel_err"].median(),
                "mu_mean_abs_err_median": d["mu_mean_abs_err"].median(),
                "x_max_abs_err_median": d["x_max_abs_err"].median(),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tables", "ablation.csv"), index=False)

    rows = []
    for version in VERSIONS:
        for key in ["analytic", "deqn", "gp"]:
            x = uq[version]["samples"][key]
            q = np.percentile(x, [5, 25, 50, 75, 95])
            rows.append(
                {"version": version, "method": key, "mean": x.mean(), "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]}
                | {f"S1_{p}": v for p, v in zip(PARAMETERS, uq[version]["sobol"][key]["first"])}
                | {f"ST_{p}": v for p, v in zip(PARAMETERS, uq[version]["sobol"][key]["total"])}
            )
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tables", "uq.csv"), index=False)

    rows = []
    for version in VERSIONS:
        h = solved[version].history
        rows.append(
            {
                "version": version,
                "iterations": len(h["iteration"]),
                "minutes": h["seconds"][-1] / 60,
                "gp_points": uq[version]["gp_points"],
                "uq_draws": UQ_DRAWS,
                "part_2_minutes_per_instance": PART_2_MINUTES,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tables", "cost.csv"), index=False)


def main() -> None:
    solved = {v: Solved(v) for v in VERSIONS}
    accuracies, baseline, uq = {}, {}, {}
    for version, s in solved.items():
        model = s.model
        test = model.low + (model.high - model.low) * qmc.Sobol(3, seed=7).random(TEST_DRAWS)
        accuracies[version], _, _ = accuracy(s, test)
        baseline[version], _, _ = accuracy(s, model.baseline[None, :])
        uq[version] = quantify(s)
        print(f"DICE {version} done", flush=True)

    # Ablation: DICE 2016 trained with the recursive conditions weighted by 1 / (1 - beta)
    weighted = Solved("2016", "_weighted")
    test = weighted.model.low + (weighted.model.high - weighted.model.low) * qmc.Sobol(3, seed=7).random(TEST_DRAWS)
    ablation, _, _ = accuracy(weighted, test)
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    ablation.to_csv(os.path.join(RESULTS, "csv", "test_draws_dice2016_weighted.csv"), index=False)

    write_tables(solved, accuracies, baseline, uq, ablation)
    figure_slices(solved)
    figure_accuracy(accuracies)
    figure_paths(solved)
    figure_uq(uq)
    figure_learning(solved, uq)
    print(pd.read_csv(os.path.join(RESULTS, "tables", "accuracy.csv")).to_string())
    print(pd.read_csv(os.path.join(RESULTS, "tables", "uq.csv")).to_string())
    print(pd.read_csv(os.path.join(RESULTS, "tables", "ablation.csv")).to_string())


if __name__ == "__main__":
    main()
