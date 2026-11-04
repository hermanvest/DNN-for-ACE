"""Stochastic dual dynamic programming (SDDP) for ACE.

The value function of each stage is approximated from above by cuts,
V_t(s) <= min_j alpha_j + b_j . s, and the method alternates two passes:

    forward    simulate the policy implied by the current cuts from the initial state
               (a lower bound on V_0: the value of a feasible policy)
    backward   from the last stage to the first, solve each stage at the visited state
               and add the cut given by its value and gradient (an upper bound on V_0)

until the gap between the bounds closes. The value after the last stage is zero; with a
horizon of 450 periods the truncation does not affect the first 58 periods,
even at the lowest discount rate.

With `shock_sd` > 0, log productivity gets an i.i.d. normal shock each period, observed
before the decisions. The backward pass then averages cuts over Gauss-Hermite nodes (the
exact expectation for a normal shock) and the forward pass samples the shocks, which is
SDDP proper. In ACE the shock shifts the value but not its slopes, so the optimal policy
is the same with or without it: a check of the machinery.

Cuts can be warm-started, for example from the cut network of `ace_sddp.neural`.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from ace_sddp.model import Instance
from ace_sddp.stage import StageProblem


@dataclass
class Result:
    iterations: int
    solves: int
    seconds: float
    lower: float
    upper: float
    history: list = field(default_factory=list)
    path: dict = field(default_factory=dict)
    cuts: list = field(default_factory=list)  # The final cuts of each stage


class SDDP:
    def __init__(self, inst: Instance, stage: StageProblem, config: dict, shock_sd: float = 0.0, nodes: int = 5, seed: int = 0):
        self.inst = inst
        self.stage = stage
        self.cfg = config["sddp"]
        self.T = inst.economy.horizon
        bound = float(self.cfg["value_bound"])
        # cuts[t] approximates V_t for t = 1..T; V_T = 0 exactly
        self.cuts = [np.array([[bound, 0, 0, 0, 0, 0, 0]], dtype=float) for _ in range(self.T)]
        self.cuts.append(np.zeros((1, 7)))
        # The initial bound and any warm-start cuts are provisional: the first backward
        # pass replaces them with cuts computed from the stage problems
        self.provisional = [True] * self.T + [False]
        self.shock_sd = shock_sd
        if shock_sd > 0:
            x, w = np.polynomial.hermite.hermgauss(nodes)
            self.nodes, self.weights = np.sqrt(2.0) * shock_sd * x, w / np.sqrt(np.pi)
        else:
            self.nodes, self.weights = np.zeros(1), np.ones(1)
        self.rng = np.random.default_rng(seed)
        self.solves = 0

    def warm_start(self, cuts) -> None:
        """Replaces the initial bound with given cuts (one array of rows [alpha, b] per stage t = 1..T-1).

        Predicted cuts guide the decisions of the first forward pass, but they are not
        guaranteed to lie above the true value function. Stage 0, whose cuts give the
        upper bound, therefore gets none, and the first backward pass replaces them all.
        """
        for t in range(1, self.T):
            self.cuts[t] = np.atleast_2d(cuts[t]).astype(float)

    def upper_bound(self) -> float:
        c = self.cuts[0]
        return float(np.min(c[:, 0] + c[:, 1:] @ self.inst.economy.s0))

    def forward(self) -> dict:
        e = self.inst.economy
        s = e.s0.copy()
        path = {k: [] for k in ("states", "x", "mu", "scc", "utility", "shock", "slopes_next")}
        for t in range(self.T):
            shock = self.rng.normal(0.0, self.shock_sd) if self.shock_sd > 0 else 0.0
            sol = self.stage.solve(self.inst, t, s, self.cuts[t + 1], shock)
            self.solves += 1
            path["states"].append(s)
            path["x"].append(sol.x)
            path["mu"].append(sol.mu)
            path["scc"].append(self.inst.scc(t, s, sol.mu, sol.slopes_next))
            path["utility"].append(self.inst.utility(t, s, sol.x, sol.mu, shock))
            path["shock"].append(shock)
            path["slopes_next"].append(sol.slopes_next)
            s = self.inst.step(t, s, sol.x, sol.mu, shock)
        path = {k: np.array(v) for k, v in path.items()}
        path["value"] = float(np.sum(path["utility"] * self.inst.beta ** np.arange(self.T)))
        return path

    def backward(self, states: np.ndarray) -> None:
        for t in reversed(range(self.T)):
            s = states[t]
            value, grad = 0.0, np.zeros(6)
            for shock, weight in zip(self.nodes, self.weights):
                sol = self.stage.solve(self.inst, t, s, self.cuts[t + 1], shock)
                self.solves += 1
                value += weight * sol.value
                grad += weight * sol.gradient
            cut = np.concatenate([[value - grad @ s], grad])
            if self.provisional[t]:
                self.cuts[t], self.provisional[t] = cut[None, :], False
            else:
                self.cuts[t] = np.vstack([self.cuts[t], cut])[-self.stage.max_cuts:]

    def run(self, max_iterations: int = None, tolerance: float = None) -> Result:
        max_iterations = max_iterations or int(self.cfg["max_iterations"])
        tolerance = float(self.cfg["tolerance"]) if tolerance is None else tolerance
        start = time.time()
        history = []
        for iteration in range(1, max_iterations + 1):
            path = self.forward()
            lower = path["value"]
            upper = self.upper_bound()
            gap = (upper - lower) / max(abs(lower), 1.0)
            history.append({"iteration": iteration, "lower": lower, "upper": upper, "gap": gap,
                            "solves": self.solves, "seconds": time.time() - start})
            # A stochastic forward pass gives a noisy lower bound, so only the deterministic case stops on the gap
            if self.shock_sd == 0 and gap < tolerance:
                break
            self.backward(path["states"])
        return Result(iteration, self.solves, time.time() - start, lower, upper, history, path,
                      [c.copy() for c in self.cuts])
