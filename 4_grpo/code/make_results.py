"""Figures and tables for GRPO on ACE.

Usage: python make_results.py
Reads   ../results/runs/dice{2016,2023}[tag]/ and, for comparison only, the results data
        of the corrected DEQN (../../2_revised/results/) and of the actor-critic
        (../../3_actor_critic/results/). No code is imported from those parts.
Writes  ../results/figures/, ../results/tables/, ../results/csv/
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

from ace_grpo.env import AceEnv
from ace_grpo.grpo import GRPO

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DEQN = os.path.join(HERE, "..", "..", "2_revised", "results")
ACTOR_CRITIC = os.path.join(HERE, "..", "..", "3_actor_critic", "results")
VERSIONS = ["2016", "2023"]
START_YEAR = {"2016": 2015, "2023": 2020}
HORIZON = 58  # Periods up to about 2300
VARIANTS = {"": "Process advantages", "_outcome": "Outcome advantages"}

COLOR_GRPO = "#2a78d6"
COLOR_AC = "#eb6834"
COLOR_OUTCOME = "#1baf7a"
COLOR_ANALYTIC = "#3d3d3a"
INK_SECONDARY = "#52514e"
ANALYTIC_STYLE = dict(color=COLOR_ANALYTIC, linestyle=(0, (4, 3)), linewidth=1.3)

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


def legend_below(fig, ax, ncol=3, y=-0.06):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


################ DATA ################
def evaluate(version: str, tag: str = ""):
    with open(os.path.join(HERE, "configs", f"ace_dice_{version}.yaml")) as f:
        config = yaml.safe_load(f)
    env = AceEnv(config)
    agent = GRPO(env, config)
    run = os.path.join(RESULTS, "runs", f"dice{version}{tag}")
    agent.network.load_weights(os.path.join(run, "policy.weights.h5"))
    agent.log_std.assign(np.load(os.path.join(run, "log_std.npy")))
    with open(os.path.join(run, "history.json")) as f:
        history = json.load(f)

    n = env.t_max
    s = agent.simulate(n + 1)
    a = agent.policy(s)
    t = s[:, 6].numpy().astype(int)
    mu = a["mu"].numpy()
    c = env.costates

    # Policy-implied SCC: the marginal abatement cost at the chosen abatement rate, in USD/tCO2
    theta_1 = env.exo.theta_1[t]
    E_BAU = env.E_BAU(s).numpy()
    dlogF_dE = theta_1 * env.theta_2 * mu ** (env.theta_2 - 1) / (E_BAU * (1 - theta_1 * mu**env.theta_2))
    net_output = tf.exp(env.log_net_output(s, a["mu"])).numpy()
    mac = 1000 * net_output * dlogF_dE
    scc_star = 1000 * tf.exp(env.log_net_output(s, tf.constant(env.mu_star[t], tf.float32))).numpy() * c.carbon_price

    # Analytic optimal path from the same initial state
    s_star = [env.initial_state(1)]
    for _ in range(n):
        s_star.append(env.step(s_star[-1], env.analytic_action(s_star[-1])))
    s_star = tf.concat(s_star, axis=0)

    V_policy = env.rollout_value(s[:1], agent.policy).numpy()[0]
    V_star = env.rollout_value(s[:1], env.analytic_action).numpy()[0]

    data = pd.DataFrame(
        {
            "year": START_YEAR[version] + env.timestep * np.arange(n + 1),
            "k": s[:, 0].numpy(),
            "M_1": s[:, 1].numpy(),
            "T_1": env.temperature(s[:, 4].numpy()),
            "x": a["x"].numpy(),
            "mu": mu,
            "mu_star": env.mu_star[t],
            "scc_policy": mac,
            "scc_star": scc_star,
            "path_k": s_star[:, 0].numpy(),
            "path_M_1": s_star[:, 1].numpy(),
            "path_T_1": env.temperature(s_star[:, 4].numpy()),
        }
    )
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    data.to_csv(os.path.join(RESULTS, "csv", f"trajectory_dice{version}{tag}.csv"), index=False)
    return {
        "env": env,
        "history": history,
        "data": data,
        "welfare_loss": float(np.exp((V_star - V_policy) * (1 - env.beta)) - 1),
        "std": agent.std().numpy(),
        "samples": agent.samples_per_iteration * len(history["iteration"]),
    }


def metrics(r):
    d = r["data"].iloc[:HORIZON]
    interior = d["mu_star"] < 0.99  # The MAC equals the SCC only where full abatement does not bind
    return {
        "max |x - x*|": float(np.max(np.abs(d["x"] - r["env"].costates.x))),
        "mean |mu - mu*|": float(np.mean(np.abs(d["mu"] - d["mu_star"]))),
        "max |mu - mu*|": float(np.max(np.abs(d["mu"] - d["mu_star"]))),
        "SCC rel. error": float(np.mean(np.abs(d["scc_policy"][interior] / d["scc_star"][interior] - 1))),
        "welfare loss (CE, %)": 100 * r["welfare_loss"],
        "training minutes": r["history"]["seconds"][-1] / 60,
        "environment transitions (millions)": r["samples"] / 1e6,
    }


def other_method(path: str, version: str, label: str):
    """Accuracy of the DEQN or actor-critic from their saved trajectories (data only)."""
    file = os.path.join(path, "csv", f"trajectory_dice{version}.csv")
    if not os.path.exists(file):
        return {}
    d = pd.read_csv(file).iloc[:HORIZON]
    x_star = 1 - (1 / 1.0286) ** 5 * 0.3
    out = {
        "max |x - x*|": float(np.max(np.abs(d["x"] - x_star))),
        "mean |mu - mu*|": float(np.mean(np.abs(d["mu"] - d["mu_star"]))),
        "max |mu - mu*|": float(np.max(np.abs(d["mu"] - d["mu_star"]))),
    }
    table = os.path.join(path, "tables", "comparison.csv")
    if label == "Actor-critic" and os.path.exists(table):
        t = pd.read_csv(table, index_col=0)
        out["SCC rel. error"] = float(t.loc["SCC rel. error", f"AC {version}"])
        out["welfare loss (CE, %)"] = float(t.loc["welfare loss (CE, %)", f"AC {version}"])
    return out


################ FIGURES ################
def fig_controls(res):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex="col")
    for col, v in enumerate(VERSIONS):
        d, env = res[v]["data"].iloc[:HORIZON], res[v]["env"]
        axes[0, col].plot(d["year"], np.full(len(d), env.costates.x), label="Analytic solution", **ANALYTIC_STYLE)
        axes[0, col].plot(d["year"], d["x"], color=COLOR_GRPO, label="GRPO (mean action)")
        axes[0, col].set_ylim(0.6, 0.85)
        axes[0, col].set_title(f"ACE-DICE {v}")
        axes[1, col].plot(d["year"], d["mu_star"], **ANALYTIC_STYLE)
        axes[1, col].plot(d["year"], d["mu"], color=COLOR_GRPO)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel("Year")
    axes[0, 0].set_ylabel("Consumption rate $x_t$")
    axes[1, 0].set_ylabel(r"Abatement rate $\mu_t$")
    legend_below(fig, axes[0, 0], ncol=2)
    fig.tight_layout()
    save(fig, "controls")


def fig_scc(res):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, v in zip(axes, VERSIONS):
        d = res[v]["data"].iloc[:HORIZON]
        interior = d["mu_star"] < 0.99
        ax.plot(d["year"][interior], d["scc_star"][interior], label="Analytic SCC", **ANALYTIC_STYLE)
        ax.plot(d["year"][interior], d["scc_policy"][interior], color=COLOR_GRPO, label="GRPO: marginal abatement cost")
        ax.set_title(f"ACE-DICE {v}")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("USD per tCO$_2$")
    legend_below(fig, axes[0], ncol=2, y=-0.08)
    fig.tight_layout()
    save(fig, "scc")


def fig_states(res):
    specs = [("T_1", "path_T_1", "Temperature (°C)"), ("M_1", "path_M_1", "Atmospheric carbon (GtC)"), ("k", "path_k", "Log capital")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for row, v in enumerate(VERSIONS):
        d = res[v]["data"].iloc[:HORIZON]
        for col, (name, star, label) in enumerate(specs):
            ax = axes[row, col]
            ax.plot(d["year"], d[star], label="Analytic optimal path", **ANALYTIC_STYLE)
            ax.plot(d["year"], d[name], color=COLOR_GRPO, label="GRPO")
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel("Year")
        axes[row, 0].set_ylabel(f"ACE-DICE {v}")
    legend_below(fig, axes[0, 0], ncol=2)
    fig.tight_layout()
    save(fig, "states")


def fig_learning(res, variants):
    """Abatement and consumption errors against training time, GRPO and the actor-critic."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for v, style in zip(VERSIONS, ["-", (0, (1, 1.5))]):
        h = res[v]["history"]
        ev = pd.DataFrame(h["eval"])
        minutes = np.array(h["seconds"])[ev["iteration"].to_numpy()] / 60
        axes[0].plot(minutes, ev["mu_err"], color=COLOR_GRPO, linestyle=style, label=f"GRPO, {v}")
        axes[1].plot(minutes, ev["x_err"], color=COLOR_GRPO, linestyle=style)
        ac = os.path.join(ACTOR_CRITIC, "runs", f"dice{v}", "history.json")
        if os.path.exists(ac):
            ha = json.load(open(ac))
            eva = pd.DataFrame(ha["eval"])
            minutes_ac = np.array(ha["seconds"])[eva["iteration"].to_numpy()] / 60
            axes[0].plot(minutes_ac, eva["mu_err"], color=COLOR_AC, linestyle=style, label=f"Actor-critic, {v}")
            axes[1].plot(minutes_ac, eva["x_err"], color=COLOR_AC, linestyle=style)
    axes[0].set_title(r"mean $|\mu_t - \mu_t^*|$")
    axes[1].set_title(r"max $|x_t - x^*|$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Training time (minutes)")
    legend_below(fig, axes[0], ncol=4, y=-0.1)
    fig.tight_layout()
    save(fig, "learning")


def fig_variants(variants):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    d0 = variants[""]["data"].iloc[:HORIZON]
    axes[0].plot(d0["year"], d0["mu_star"], label="Analytic solution", **ANALYTIC_STYLE)
    for tag, r in variants.items():
        color = COLOR_GRPO if tag == "" else COLOR_OUTCOME
        d = r["data"].iloc[:HORIZON]
        axes[0].plot(d["year"], d["mu"], color=color, label=VARIANTS[tag])
        ev = pd.DataFrame(r["history"]["eval"])
        axes[1].plot(ev["iteration"], ev["mu_err"], color=color)
    axes[0].set_title(r"Abatement rate $\mu_t$ (ACE-DICE 2016)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_title(r"mean $|\mu_t - \mu_t^*|$ during training")
    axes[1].set_xlabel("Iteration")
    axes[1].set_yscale("log")
    legend_below(fig, axes[0], ncol=3, y=-0.08)
    fig.tight_layout()
    save(fig, "advantages")


################ TABLES ################
def write_tables(res, variants):
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    rows = {}
    for v in VERSIONS:
        rows[f"GRPO {v}"] = metrics(res[v])
        rows[f"Actor-critic {v}"] = other_method(ACTOR_CRITIC, v, "Actor-critic")
        rows[f"DEQN {v}"] = other_method(DEQN, v, "DEQN")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "tables", "comparison.csv"))
    pd.DataFrame({VARIANTS[t]: metrics(r) for t, r in variants.items()}).to_csv(
        os.path.join(RESULTS, "tables", "advantages.csv")
    )
    print(table.to_string())
    print(pd.read_csv(os.path.join(RESULTS, "tables", "advantages.csv"), index_col=0).to_string())


def main():
    res = {v: evaluate(v) for v in VERSIONS}
    variants = {"": res["2016"]}
    if os.path.exists(os.path.join(RESULTS, "runs", "dice2016_outcome", "policy.weights.h5")):
        variants["_outcome"] = evaluate("2016", "_outcome")
    fig_controls(res)
    fig_scc(res)
    fig_states(res)
    fig_learning(res, variants)
    fig_variants(variants)
    write_tables(res, variants)


if __name__ == "__main__":
    main()
