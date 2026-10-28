"""Figures and tables for the actor-critic paper.

Usage: python make_results.py
Reads   ../results/runs/dice{2016,2023}[tag]/ and, for comparison only, the trajectory
        CSVs of the corrected DEQN in ../../revised/results/csv/ (data, no code).
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

from ace_ac.agent import ActorCritic
from ace_ac.env import AceEnv

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DEQN_CSV = os.path.join(HERE, "..", "..", "revised", "results", "csv")
VERSIONS = ["2016", "2023"]
START_YEAR = {"2016": 2015, "2023": 2020}
HORIZON = 58  # Periods up to about 2300
ABLATIONS = {"": "Full method", "_td_only": "TD learning only", "_no_exploration": "No state perturbation"}

COLOR_AC = "#2a78d6"
COLOR_ANALYTIC = "#3d3d3a"
COLOR_ABLATION = {"": "#2a78d6", "_td_only": "#eb6834", "_no_exploration": "#1baf7a"}
INK_SECONDARY = "#52514e"

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
ANALYTIC_STYLE = dict(color=COLOR_ANALYTIC, linestyle=(0, (4, 3)), linewidth=1.3)


def save(fig, name):
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "figures", f"{name}.pdf"))
    fig.savefig(os.path.join(RESULTS, "figures", f"{name}.png"))
    plt.close(fig)


def legend_below(fig, ax, ncol=3, y=-0.06):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y))


################ DATA ################
def load(version: str, tag: str = ""):
    with open(os.path.join(HERE, "configs", f"ace_dice_{version}.yaml")) as f:
        config = yaml.safe_load(f)
    env = AceEnv(config)
    agent = ActorCritic(env, config)
    run = os.path.join(RESULTS, "runs", f"dice{version}{tag}")
    agent.actor.load_weights(os.path.join(run, "actor.weights.h5"))
    agent.critic.load_weights(os.path.join(run, "critic.weights.h5"))
    with open(os.path.join(run, "history.json")) as f:
        history = json.load(f)
    return env, agent, history


def evaluate(version: str, tag: str = ""):
    env, agent, history = load(version, tag)
    c = env.costates
    n = env.t_max
    s = agent.simulate(n + 1)
    a = agent.policy(s)
    s_next = env.step(s, a)
    lam = agent.costates(s_next).numpy()
    t = s[:, 6].numpy().astype(int)

    # Analytic optimal path from the same initial state
    s_star = [env.initial_state(1)]
    for _ in range(n):
        s_star.append(env.step(s_star[-1], env.analytic_action(s_star[-1])))
    s_star = tf.concat(s_star, axis=0)

    # Values: critic, true value of the learned policy, and the optimum, at the policy's states
    V_critic = agent.value(s).numpy()
    V_policy = env.rollout_value(s, agent.policy).numpy()
    V_star = env.rollout_value(s, env.analytic_action).numpy()
    welfare_loss = float(np.exp((V_star[0] - V_policy[0]) * (1 - env.beta)) - 1)

    mu = a["mu"].numpy()
    data = pd.DataFrame(
        {
            "year": START_YEAR[version] + env.timestep * np.arange(n + 1),
            "k": s[:, 0].numpy(),
            "M_1": s[:, 1].numpy(),
            "T_1": env.temperature(s[:, 4].numpy()),
            "x": a["x"].numpy(),
            "mu": mu,
            "mu_star": env.mu_star[t],
            "lambda_k": lam[:, 0],
            "lambda_M_1": lam[:, 1],
            "lambda_tau_1": lam[:, 4],
            "scc": env.scc(s, a["mu"], lam[:, 0], lam[:, 1]).numpy(),
            "scc_star": env.scc(
                s, tf.constant(env.mu_star[t], tf.float32), tf.fill([n + 1], c.lambda_k), tf.fill([n + 1], c.lambda_M[0])
            ).numpy(),
            "V_critic": V_critic,
            "V_policy": V_policy,
            "V_star": V_star,
            "path_k": s_star[:, 0].numpy(),
            "path_M_1": s_star[:, 1].numpy(),
            "path_T_1": env.temperature(s_star[:, 4].numpy()),
        }
    )
    os.makedirs(os.path.join(RESULTS, "csv"), exist_ok=True)
    data.to_csv(os.path.join(RESULTS, "csv", f"trajectory_dice{version}{tag}.csv"), index=False)
    return {"env": env, "history": history, "data": data, "welfare_loss": welfare_loss}


def deqn(version: str):
    path = os.path.join(DEQN_CSV, f"trajectory_dice{version}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


################ FIGURES ################
def fig_controls(res):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex="col")
    for col, v in enumerate(VERSIONS):
        d, env = res[v]["data"].iloc[:HORIZON], res[v]["env"]
        axes[0, col].plot(d["year"], np.full(len(d), env.costates.x), label="Analytic solution", **ANALYTIC_STYLE)
        axes[0, col].plot(d["year"], d["x"], color=COLOR_AC, label="Actor-critic")
        axes[0, col].set_ylim(0.6, 0.85)
        axes[0, col].set_title(f"ACE-DICE {v}")
        axes[1, col].plot(d["year"], d["mu_star"], **ANALYTIC_STYLE)
        axes[1, col].plot(d["year"], d["mu"], color=COLOR_AC)
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
        ax.plot(d["year"], d["scc_star"], label="Analytic solution", **ANALYTIC_STYLE)
        ax.plot(d["year"], d["scc"], color=COLOR_AC, label=r"Actor-critic ($\beta\nabla V_\phi$)")
        ax.set_title(f"ACE-DICE {v}")
        ax.set_xlabel("Year")
    axes[0].set_ylabel("SCC (USD per tCO$_2$)")
    legend_below(fig, axes[0], ncol=2, y=-0.08)
    fig.tight_layout()
    save(fig, "scc")


def fig_costates(res):
    specs = [
        ("lambda_k", lambda c: c.lambda_k, r"$\lambda_{k}$"),
        ("lambda_M_1", lambda c: c.lambda_M[0], r"$\lambda_{M_1}$"),
        ("lambda_tau_1", lambda c: c.lambda_tau[0], r"$\lambda_{\tau_1}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (name, value, label) in zip(axes, specs):
        for v, style in zip(VERSIONS, ["-", (0, (1, 1.5))]):
            d = res[v]["data"].iloc[:HORIZON]
            ax.plot(d["year"], d[name] / value(res[v]["env"].costates), color=COLOR_AC, linestyle=style, label=f"ACE-DICE {v}")
        ax.axhline(1.0, **ANALYTIC_STYLE, label="Analytic solution")
        ax.set_title(label)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Implied / analytic costate")
    legend_below(fig, axes[0], ncol=3, y=-0.1)
    fig.tight_layout()
    save(fig, "costates")


def fig_states(res):
    specs = [("T_1", "path_T_1", "Temperature (°C)"), ("M_1", "path_M_1", "Atmospheric carbon (GtC)"), ("k", "path_k", "Log capital")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for row, v in enumerate(VERSIONS):
        d = res[v]["data"].iloc[:HORIZON]
        for col, (name, star, label) in enumerate(specs):
            ax = axes[row, col]
            ax.plot(d["year"], d[star], label="Analytic optimal path", **ANALYTIC_STYLE)
            ax.plot(d["year"], d[name], color=COLOR_AC, label="Actor-critic")
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel("Year")
        axes[row, 0].set_ylabel(f"ACE-DICE {v}")
    legend_below(fig, axes[0, 0], ncol=2)
    fig.tight_layout()
    save(fig, "states")


def fig_learning(res):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    warmup = res["2016"]["env"].config["training"]["critic_warmup"]
    for v, style in zip(VERSIONS, ["-", (0, (1, 1.5))]):
        ev = pd.DataFrame(res[v]["history"]["eval"])
        axes[0].plot(ev["iteration"], ev["x_err"], color=COLOR_AC, linestyle=style, label=f"ACE-DICE {v}")
        axes[1].plot(ev["iteration"], ev["mu_err"], color=COLOR_AC, linestyle=style)
        axes[2].plot(ev["iteration"], ev["lambda_M1_ratio"], color=COLOR_AC, linestyle=style)
    axes[0].set_title(r"max $|x_t - x^*|$")
    axes[1].set_title(r"mean $|\mu_t - \mu_t^*|$")
    axes[2].set_title(r"$\lambda_{M_1}$ implied / analytic")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].axhline(1.0, **ANALYTIC_STYLE)
    axes[2].set_ylim(-3, 2)
    for ax in axes:
        ax.axvspan(0, warmup, color="#f0efec", zorder=0)
        ax.set_xlabel("Iteration")
    legend_below(fig, axes[0], ncol=2, y=-0.1)
    fig.tight_layout()
    save(fig, "learning")


def fig_ablation(ablations):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    first = next(iter(ablations.values()))
    d0 = first["data"].iloc[:HORIZON]
    axes[0].plot(d0["year"], d0["mu_star"], label="Analytic solution", **ANALYTIC_STYLE)
    for tag, r in ablations.items():
        d = r["data"].iloc[:HORIZON]
        axes[0].plot(d["year"], d["mu"], color=COLOR_ABLATION[tag], label=ABLATIONS[tag])
        ev = pd.DataFrame(r["history"]["eval"])
        axes[1].plot(ev["iteration"], ev["mu_err"], color=COLOR_ABLATION[tag])
    axes[0].set_title(r"Abatement rate $\mu_t$ (ACE-DICE 2016)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_title(r"mean $|\mu_t - \mu_t^*|$ during training")
    axes[1].set_xlabel("Iteration")
    axes[1].set_yscale("log")
    legend_below(fig, axes[0], ncol=4, y=-0.08)
    fig.tight_layout()
    save(fig, "ablation")


################ TABLES ################
def metrics(r, version):
    d, env = r["data"].iloc[:HORIZON], r["env"]
    c = env.costates
    rel = lambda a, b: float(np.mean(np.abs(a / b - 1)))
    return {
        "max |x - x*|": float(np.max(np.abs(d["x"] - c.x))),
        "mean |mu - mu*|": float(np.mean(np.abs(d["mu"] - d["mu_star"]))),
        "max |mu - mu*|": float(np.max(np.abs(d["mu"] - d["mu_star"]))),
        "lambda_k rel. error": rel(d["lambda_k"], c.lambda_k),
        "lambda_M1 rel. error": rel(d["lambda_M_1"], c.lambda_M[0]),
        "lambda_tau1 rel. error": rel(d["lambda_tau_1"], c.lambda_tau[0]),
        "SCC rel. error": rel(d["scc"], d["scc_star"]),
        "SCC 2050 (AC)": float(np.interp(2050, d["year"], d["scc"])),
        "SCC 2050 (analytic)": float(np.interp(2050, d["year"], d["scc_star"])),
        "critic rel. error vs V^pi": rel(d["V_critic"], d["V_policy"]),
        "welfare loss (CE, %)": 100 * r["welfare_loss"],
    }


def deqn_metrics(version):
    d = deqn(version)
    if d is None:
        return {}
    d = d.iloc[:HORIZON]
    x_star = 1 - (1 / 1.0286) ** 5 * 0.3
    return {
        "max |x - x*|": float(np.max(np.abs(d["x"] - x_star))),
        "mean |mu - mu*|": float(np.mean(np.abs(d["mu"] - d["mu_star"]))),
        "max |mu - mu*|": float(np.max(np.abs(d["mu"] - d["mu_star"]))),
        "SCC rel. error": float(np.mean(np.abs(d["scc"] / d["scc_star"] - 1))),
    }


def write_tables(res, ablations):
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    rows = {}
    for v in VERSIONS:
        rows[f"AC {v}"] = metrics(res[v], v)
        rows[f"DEQN {v}"] = deqn_metrics(v)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "tables", "comparison.csv"))
    ab = pd.DataFrame({ABLATIONS[tag]: metrics(r, "2016") for tag, r in ablations.items()})
    ab.to_csv(os.path.join(RESULTS, "tables", "ablation.csv"))

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--"
        return f"{100 * x:.2f}\\%" if pct else (f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}")

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{ACE-DICE 2016} & \multicolumn{2}{c}{ACE-DICE 2023} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & Actor-critic & DEQN & Actor-critic & DEQN \\",
        r"\midrule",
    ]
    spec = [
        (r"$\max_t |x_t - x^*|$", "max |x - x*|", False),
        (r"mean $|\mu_t - \mu_t^*|$", "mean |mu - mu*|", False),
        (r"$\max_t |\mu_t - \mu_t^*|$", "max |mu - mu*|", False),
        (r"SCC, mean relative error", "SCC rel. error", True),
        (r"$\lambda_k$, mean relative error", "lambda_k rel. error", True),
        (r"$\lambda_{M_1}$, mean relative error", "lambda_M1 rel. error", True),
        (r"$\lambda_{\tau_1}$, mean relative error", "lambda_tau1 rel. error", True),
        (r"Welfare loss (consumption equivalent)", "welfare loss (CE, %)", False),
    ]
    for label, key, pct in spec:
        cells = []
        for v in VERSIONS:
            for col in (f"AC {v}", f"DEQN {v}"):
                val = table[col].get(key, np.nan)
                if key == "welfare loss (CE, %)" and not np.isnan(val):
                    cells.append(f"{val:.4f}\\%")
                else:
                    cells.append(fmt(val, pct))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS, "tables", "comparison.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{l" + "c" * len(ab.columns) + "}", r"\toprule", " & " + " & ".join(ab.columns) + r" \\", r"\midrule"]
    for label, key, pct in spec:
        vals = []
        for col in ab.columns:
            val = ab[col][key]
            vals.append(f"{val:.4f}\\%" if key == "welfare loss (CE, %)" else fmt(val, pct))
        lines.append(label + " & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(RESULTS, "tables", "ablation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(table.to_string())
    print(ab.to_string())


def main():
    res = {v: evaluate(v) for v in VERSIONS}
    ablations = {}
    for tag in ABLATIONS:
        if os.path.exists(os.path.join(RESULTS, "runs", f"dice2016{tag}", "actor.weights.h5")):
            ablations[tag] = res["2016"] if tag == "" else evaluate("2016", tag)
    fig_controls(res)
    fig_scc(res)
    fig_costates(res)
    fig_states(res)
    fig_learning(res)
    fig_ablation(ablations)
    write_tables(res, ablations)


if __name__ == "__main__":
    main()
