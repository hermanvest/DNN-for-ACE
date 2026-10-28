"""Generate the figures and tables of the revised thesis from trained networks.

Usage: python make_results.py
Reads   ../results/runs/dice{2016,2023}/{policy.weights.h5,history.json}
Writes  ../results/figures/*.pdf|png, ../results/tables/*.tex, ../results/csv/*.csv
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

from ace_deqn.analytic import analytic_actions, analytic_costates, analytic_trajectory
from ace_deqn.model import AceDice, STATE_NAMES
from ace_deqn.network import build_policy_network
from ace_deqn.residuals import RESIDUAL_NAMES, residuals

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
ORIGINAL_RUNS = os.path.join(HERE, "..", "..", "plotting", "plots", "model_run")
VERSIONS = ["2016", "2023"]
START_YEAR = {"2016": 2015, "2023": 2020}
PLOT_PERIODS = 58  # Show the paths up to roughly 2300

# Validated categorical slots (blue, orange) plus neutral ink for the benchmark
COLOR_DEQN = "#2a78d6"
COLOR_ORIGINAL = "#eb6834"
COLOR_ANALYTIC = "#3d3d3a"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": "#0b0b0b",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def save(fig, name: str) -> None:
    out = os.path.join(RESULTS, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"))
    plt.close(fig)


def legend_below(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.06))


################ DATA ################
def load(version: str):
    with open(os.path.join(HERE, "configs", f"ace_dice_{version}.yaml")) as f:
        config = yaml.safe_load(f)
    model = AceDice(config)
    network = build_policy_network(config)
    run_dir = os.path.join(RESULTS, "runs", f"dice{version}")
    network.load_weights(os.path.join(run_dir, "policy.weights.h5"))
    with open(os.path.join(run_dir, "history.json")) as f:
        history = json.load(f)
    return model, network, history


def simulate(model: AceDice, network: tf.keras.Model):
    """Network trajectory with T + 1 states, so that every period has a successor."""
    s = model.initial_state(1)
    states = []
    for _ in range(model.t_max + 1):
        states.append(s)
        a = model.decode_actions(s, network(model.preprocess(s)))
        s = model.step(s, a)
    s = tf.concat(states, axis=0)
    a = model.decode_actions(s, network(model.preprocess(s)))
    return s, a


def original_run(version: str):
    path = os.path.join(ORIGINAL_RUNS, f"ACE_DICE{version}", "csv", "trajectory_data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # The original network output raw emissions; the share of E_BAU is their sigmoid
    df["mu"] = 1 - 1 / (1 + np.exp(-df["E_t"]))
    return df


def collect(version: str):
    model, network, history = load(version)
    s, a = simulate(model, network)
    n = model.t_max
    r = residuals(
        model,
        s[:n],
        {k: v[:n] for k, v in a.items()},
        s[1:],
        {k: v[1:] for k, v in a.items()},
    ).numpy()

    star = analytic_costates(model)
    a_star = analytic_actions(model, s)
    path = analytic_trajectory(model, n + 1)
    s_star = tf.constant(path["states"], tf.float32)
    a_path = analytic_actions(model, s_star)

    E_BAU = model.E_BAU(s).numpy()
    years = START_YEAR[version] + model.timestep * np.arange(n + 1)
    data = pd.DataFrame(
        {
            "year": years,
            **{name: s[:, i].numpy() for i, name in enumerate(STATE_NAMES)},
            "x": a["x"].numpy(),
            "E": a["E"].numpy(),
            "E_BAU": E_BAU,
            "mu": 1 - a["E"].numpy() / E_BAU,
            "V": a["V"].numpy(),
            "lambda_k": a["lambda_k"].numpy(),
            "lambda_M_1": a["lambda_M"][:, 0].numpy(),
            "lambda_M_2": a["lambda_M"][:, 1].numpy(),
            "lambda_M_3": a["lambda_M"][:, 2].numpy(),
            "lambda_tau_1": a["lambda_tau"][:, 0].numpy(),
            "lambda_tau_2": a["lambda_tau"][:, 1].numpy(),
            "lambda_BAU": a["lambda_BAU"].numpy(),
            "lambda_E": a["lambda_E"].numpy(),
            "scc": model.social_cost_of_carbon(s, a).numpy(),
            "T_1": model.temperature(s[:, 4].numpy()),
            "mu_star": 1 - a_star["E"].numpy() / E_BAU,
            "scc_star": model.social_cost_of_carbon(s, a_star).numpy(),
            # The analytic optimum simulated from the same initial state
            "path_k": path["states"][:, 0],
            "path_M_1": path["states"][:, 1],
            "path_T_1": model.temperature(path["states"][:, 4]),
            "path_mu": 1 - a_path["E"].numpy() / model.E_BAU(s_star).numpy(),
            "path_scc": model.social_cost_of_carbon(s_star, a_path).numpy(),
            "path_V": path["V"],
        }
    )
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    data.to_csv(os.path.join(RESULTS, "csv", f"trajectory_dice{version}.csv"), index=False)
    return {
        "model": model,
        "history": history,
        "data": data,
        "residuals": r,
        "star": star,
        "original": original_run(version),
    }


################ FIGURES ################
def fig_training(results):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, version in zip(axes, VERSIONS):
        h = results[version]["history"]
        loss = np.array(h["loss"])
        smooth = pd.Series(loss).rolling(25, min_periods=1).median()
        ax.plot(h["iteration"], loss, color=COLOR_DEQN, alpha=0.25, linewidth=0.8)
        ax.plot(h["iteration"], smooth, color=COLOR_DEQN, label="Policy objective (rolling median)")
        ax.set_yscale("log")
        ax.set_title(f"ACE-DICE {version}")
        ax.set_xlabel("Iteration")
    axes[0].set_ylabel("Mean squared residual")
    legend_below(fig, axes[0], ncol=1)
    fig.tight_layout()
    save(fig, "training_loss")


def _paths(ax, years, deqn, analytic, original=None, analytic_label="Analytic solution"):
    k = PLOT_PERIODS
    if original is not None:
        ax.plot(years[:k], original[:k], color=COLOR_ORIGINAL, label="Original implementation")
    ax.plot(years[:k], analytic[:k], color=COLOR_ANALYTIC, linestyle=(0, (4, 3)), linewidth=1.3, label=analytic_label)
    ax.plot(years[:k], deqn[:k], color=COLOR_DEQN, label="Corrected DEQN")


def fig_controls(results):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    for col, version in enumerate(VERSIONS):
        res = results[version]
        d, orig = res["data"], res["original"]
        x_star = np.full(len(d), res["star"].x)
        _paths(axes[0, col], d["year"], d["x"], x_star, None if orig is None else orig["x_t"].to_numpy())
        axes[0, col].set_title(f"ACE-DICE {version}")
        axes[0, col].set_ylim(0, 1.05)
        _paths(axes[1, col], d["year"], d["mu"], d["mu_star"], None if orig is None else orig["mu"].to_numpy())
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel("Year")
    axes[0, 0].set_ylabel("Consumption rate $x_t$")
    axes[1, 0].set_ylabel(r"Abatement rate $\mu_t$")
    legend_below(fig, axes[0, 0])
    fig.tight_layout()
    save(fig, "controls")


def fig_scc(results):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, version in zip(axes, VERSIONS):
        d = results[version]["data"]
        _paths(ax, d["year"], d["scc"], d["scc_star"])
        ax.set_title(f"ACE-DICE {version}")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("SCC (USD per tCO$_2$)")
    legend_below(fig, axes[0], ncol=2)
    fig.tight_layout()
    save(fig, "scc")


def fig_costates(results):
    specs = [
        ("lambda_k", lambda c: c.lambda_k, r"$\lambda_{k,t}$"),
        ("lambda_M_1", lambda c: c.lambda_M[0], r"$\lambda_{M_1,t}$"),
        ("lambda_tau_1", lambda c: c.lambda_tau[0], r"$\lambda_{\tau_1,t}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for row, version in enumerate(VERSIONS):
        res = results[version]
        d = res["data"]
        for col, (name, value, label) in enumerate(specs):
            ax = axes[row, col]
            _paths(ax, d["year"], d[name], np.full(len(d), value(res["star"])))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel("Year")
        axes[row, 0].set_ylabel(f"ACE-DICE {version}")
    legend_below(fig, axes[0, 0], ncol=2)
    fig.tight_layout()
    save(fig, "costates")


def fig_states(results):
    specs = [
        ("T_1", "path_T_1", "Atmospheric temperature (°C)"),
        ("M_1", "path_M_1", "Atmospheric carbon (GtC)"),
        ("k", "path_k", "Log capital $k_t$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for row, version in enumerate(VERSIONS):
        d = results[version]["data"]
        for col, (deqn, analytic, label) in enumerate(specs):
            ax = axes[row, col]
            _paths(ax, d["year"], d[deqn], d[analytic], analytic_label="Analytic optimal path")
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel("Year")
        axes[row, 0].set_ylabel(f"ACE-DICE {version}")
    legend_below(fig, axes[0, 0], ncol=2)
    fig.tight_layout()
    save(fig, "states")


def fig_residuals(results):
    groups = {
        r"$\ell_0$ capital": [0],
        r"$\ell_1$ carbon": [1, 2, 3],
        r"$\ell_2$ temperature": [4, 5],
        r"$\ell_3$ consumption": [6],
        r"$\ell_4$ emissions": [7],
        r"$\ell_6$ Bellman": [10],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, version in zip(axes, VERSIONS):
        r = np.abs(results[version]["residuals"])
        # Reversed so that l0 is drawn at the top
        box = [r[:, idx].max(axis=1) for idx in reversed(list(groups.values()))]
        parts = ax.boxplot(box, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.55)
        for patch in parts["boxes"]:
            patch.set_facecolor("#dbe8f8")
            patch.set_edgecolor(COLOR_DEQN)
        for key in ("whiskers", "caps", "medians"):
            for line in parts[key]:
                line.set_color(COLOR_DEQN if key != "medians" else "#0b0b0b")
        ax.set_yticks(range(1, len(groups) + 1), list(reversed(list(groups.keys()))))
        ax.set_xscale("log")
        ax.set_title(f"ACE-DICE {version}")
        ax.set_xlabel("|residual| over the episode (scaled units)")
    fig.tight_layout()
    save(fig, "residuals")


################ TABLES ################
def table_residuals(results):
    rows = []
    for version in VERSIONS:
        r = np.abs(results[version]["residuals"])
        for name, col in zip(RESIDUAL_NAMES, r.T):
            rows.append(
                {
                    "version": version,
                    "condition": name,
                    "mean": col.mean(),
                    "median": np.median(col),
                    "p99.9": np.quantile(col, 0.999),
                    "max": col.max(),
                }
            )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "tables", "residual_statistics.csv"), index=False)
    return df


def table_comparison(results):
    rows = []
    for version in VERSIONS:
        res = results[version]
        d, star = res["data"], res["star"]
        k = PLOT_PERIODS
        rows.append(
            {
                "version": version,
                "x_deqn_mean": d["x"][:k].mean(),
                "x_star": star.x,
                "x_max_abs_err": np.abs(d["x"][:k] - star.x).max(),
                "mu_mean_abs_err": np.abs(d["mu"][:k] - d["mu_star"][:k]).mean(),
                "mu_max_abs_err": np.abs(d["mu"][:k] - d["mu_star"][:k]).max(),
                "scc_2050_deqn": np.interp(2050, d["year"], d["scc"]),
                "scc_2050_star": np.interp(2050, d["year"], d["scc_star"]),
                "lambda_k_deqn": d["lambda_k"][:k].mean(),
                "lambda_k_star": star.lambda_k,
                "lambda_M1_deqn": d["lambda_M_1"][:k].mean(),
                "lambda_M1_star": star.lambda_M[0],
                "lambda_tau1_deqn": d["lambda_tau_1"][:k].mean(),
                "lambda_tau1_star": star.lambda_tau[0],
                "T_2100_deqn": np.interp(2100, d["year"], d["T_1"]),
                "T_2100_star": np.interp(2100, d["year"], d["path_T_1"]),
                "T_max_deqn": d["T_1"][:k].max(),
                "T_max_star": d["path_T_1"][:k].max(),
                "original_x_mean": None if res["original"] is None else res["original"]["x_t"][:k].mean(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "tables", "comparison.csv"), index=False)
    return df


LATEX_CONDITIONS = [
    r"$\ell_0$",
    r"$\ell_{1,M_1}$",
    r"$\ell_{1,M_2}$",
    r"$\ell_{1,M_3}$",
    r"$\ell_{2,\tau_1}$",
    r"$\ell_{2,\tau_2}$",
    r"$\ell_3$",
    r"$\ell_4$",
    r"$\ell_{5,A}$",
    r"$\ell_{5,B}$",
    r"$\ell_6$",
]


def latex_residual_table(results, version: str) -> None:
    """Rows are statistics, columns are conditions, as in the original thesis tables."""
    r = np.abs(results[version]["residuals"])
    stats = {
        "mean": r.mean(axis=0),
        "std": r.std(axis=0),
        "min": r.min(axis=0),
        "25\\%": np.quantile(r, 0.25, axis=0),
        "50\\%": np.quantile(r, 0.5, axis=0),
        "75\\%": np.quantile(r, 0.75, axis=0),
        "99.9\\%": np.quantile(r, 0.999, axis=0),
        "max": r.max(axis=0),
    }
    lines = [
        "\\begin{tabular}{l" + "r" * len(LATEX_CONDITIONS) + "}",
        "\\toprule",
        "Statistic & " + " & ".join(LATEX_CONDITIONS) + " \\\\",
        "\\midrule",
    ]
    for name, values in stats.items():
        lines.append(name + " & " + " & ".join(f"{v:.1e}" for v in values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(RESULTS, "tables", f"residuals_dice{version}.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def latex_comparison_table(comparison: pd.DataFrame) -> None:
    c = comparison.set_index("version")
    rows = [
        ("Consumption rate $x$ (mean to 2300)", "x_deqn_mean", "x_star", "{:.4f}"),
        (r"Costate $\lambda_k$", "lambda_k_deqn", "lambda_k_star", "{:.4f}"),
        (r"Costate $\lambda_{M_1}$ ($\times 10^{-4}$)", "lambda_M1_deqn", "lambda_M1_star", "{:.3f}"),
        (r"Costate $\lambda_{\tau_1}$", "lambda_tau1_deqn", "lambda_tau1_star", "{:.4f}"),
        ("SCC in 2050 (USD/tCO$_2$)", "scc_2050_deqn", "scc_2050_star", "{:.1f}"),
        ("Temperature in 2100 (°C)", "T_2100_deqn", "T_2100_star", "{:.2f}"),
        ("Peak temperature (°C)", "T_max_deqn", "T_max_star", "{:.2f}"),
    ]
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{ACE-DICE 2016} & \\multicolumn{2}{c}{ACE-DICE 2023} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        " & DEQN & Analytic & DEQN & Analytic \\\\",
        "\\midrule",
    ]
    for label, deqn, star, fmt in rows:
        scale = 1e4 if "M1" in deqn else 1.0
        cells = []
        for version in VERSIONS:
            cells += [fmt.format(c.loc[version, deqn] * scale), fmt.format(c.loc[version, star] * scale)]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    mu = [f"{c.loc[v, 'mu_mean_abs_err']:.4f}" for v in VERSIONS]
    mu_max = [f"{c.loc[v, 'mu_max_abs_err']:.4f}" for v in VERSIONS]
    lines += [
        "\\midrule",
        r"Abatement rate, mean $|\mu - \mu^*|$ & \multicolumn{2}{c}{" + mu[0] + "} & \\multicolumn{2}{c}{" + mu[1] + "} \\\\",
        r"Abatement rate, max $|\mu - \mu^*|$ & \multicolumn{2}{c}{" + mu_max[0] + "} & \\multicolumn{2}{c}{" + mu_max[1] + "} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(os.path.join(RESULTS, "tables", "comparison.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    results = {version: collect(version) for version in VERSIONS}
    fig_training(results)
    fig_controls(results)
    fig_scc(results)
    fig_costates(results)
    fig_states(results)
    fig_residuals(results)
    print(table_residuals(results).to_string())
    comparison = table_comparison(results)
    print(comparison.T.to_string())
    for version in VERSIONS:
        latex_residual_table(results, version)
    latex_comparison_table(comparison)


if __name__ == "__main__":
    main()
