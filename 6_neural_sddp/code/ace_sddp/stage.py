"""The convex problem of one stage, with the next value function approximated by cuts.

    V_t(s) = max_{x, mu}  log x + g_t(s, mu) + beta * theta
             s.t.         theta <= alpha_j + b_j . s'(x, mu)   for every cut j

Next capital is concave in (x, mu), carbon is affine in mu and the climate is fixed, so
with b_j,k >= 0 (more capital is worth more) every cut constraint is convex. The problem
is built once in cvxpy's parametrised form (DPP) and re-solved with new parameters, so a
solve takes about a millisecond.

The gradient of V_t with respect to the incoming state follows from the envelope theorem,
using the multipliers pi_j of the cut constraints (they sum to beta):

    dV/dk   = kappa (1 + w_k),                      w_k = sum_j pi_j b_j,k
    dV/dM   = Phi^T sum_j pi_j b_j,M + (sigma_forc / M_pre) sum_j pi_j b_j,tau1 e_1
    dV/dtau = -xi_0 (1 + w_k) e_1 + sigma^T sum_j pi_j b_j,tau

This is the costate recursion of the DEQN residuals in parts 2 and 5, computed from duals.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from ace_sddp.model import Instance

Z_FLOOR = -60.0  # Lower bound on the concave part of next capital, for cuts that ignore capital


@dataclass
class StageSolution:
    x: float
    mu: float
    value: float  # V_t at the incoming state, including the constant part of the reward
    gradient: np.ndarray  # dV_t / ds
    slopes_next: np.ndarray  # Multiplier-weighted slope of V_t+1 at the solution
    status: str


class StageProblem:
    def __init__(self, max_cuts: int, consumption_bounds=(0.2, 0.98)):
        self.max_cuts = max_cuts
        self.x, self.mu, self.theta = cp.Variable(), cp.Variable(), cp.Variable()
        self.w, self.z = cp.Variable(), cp.Variable()  # w <= log(1 - theta_1 mu^theta_2), z <= w + log(1 - x)
        self.beta = cp.Parameter(nonneg=True)
        self.theta_1 = cp.Parameter(nonneg=True)
        self.const = cp.Parameter(max_cuts)
        self.slope_k = cp.Parameter(max_cuts, nonneg=True)
        self.slope_mu = cp.Parameter(max_cuts)
        self.theta_2 = None
        self.bounds = consumption_bounds
        self.problem = None

    def _build(self, theta_2: float) -> None:
        self.theta_2 = theta_2
        lo, hi = self.bounds
        self.cuts = self.theta <= self.const + self.slope_k * self.z + self.slope_mu * self.mu
        constraints = [
            self.cuts,
            self.w <= cp.log(1 - self.theta_1 * cp.power(self.mu, theta_2)),
            self.z <= self.w + cp.log(1 - self.x), self.z >= Z_FLOOR,
            self.x >= lo, self.x <= hi, self.mu >= 0, self.mu <= 1,
        ]
        self.problem = cp.Problem(cp.Maximize(cp.log(self.x) + self.w + self.beta * self.theta), constraints)

    def solve(self, inst: Instance, t: int, s: np.ndarray, cuts: np.ndarray, shock: float = 0.0) -> StageSolution:
        """Solves stage t at state s, with cuts of V_t+1 as rows [alpha, b_k, b_M1..3, b_tau1..2]."""
        e = inst.economy
        if self.problem is None or self.theta_2 != e.theta_2:
            self._build(e.theta_2)
        cuts = cuts[-self.max_cuts:]
        alpha, b = cuts[:, 0], cuts[:, 1:]
        b_k = np.maximum(b[:, 0], 0.0)

        g0 = e.log_Y_gross(t, s[0]) + shock - inst.xi_0 * s[4] + inst.xi_0
        M_fixed = inst.Phi @ s[1:4]
        M_fixed[0] += e.zeta * (e.E_BAU[t] + e.E_land[t])
        tau_next = e.sigma @ s[4:6]
        tau_next[0] += e.sigma_forc * s[1] / e.M_pre

        # Unused slots repeat the last cut: a duplicate constraint changes nothing
        n = len(cuts)
        pad = np.r_[np.arange(n), np.full(self.max_cuts - n, n - 1)]
        const = (alpha + b_k * (g0 + e.log_capital_factor) + b[:, 1:4] @ M_fixed + b[:, 4:6] @ tau_next)[pad]
        slope_k = b_k[pad]
        slope_mu = (-b[:, 1] * e.zeta * e.E_BAU[t])[pad]

        self.beta.value = inst.beta
        self.theta_1.value = float(e.theta_1[t])
        self.const.value, self.slope_k.value, self.slope_mu.value = const, slope_k, slope_mu
        try:
            self.problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            self.problem.solve(solver=cp.SCS, eps=1e-9, max_iters=20000)

        pi = np.bincount(pad, weights=np.maximum(np.asarray(self.cuts.dual_value, dtype=float), 0.0), minlength=n)
        pi *= inst.beta / max(pi.sum(), 1e-12)  # Exact up to solver tolerance
        weighted = pi @ b  # sum_j pi_j b_j, with the k entry using the clipped slopes
        weighted[0] = pi @ b_k
        w_k = weighted[0]
        grad = np.zeros(6)
        grad[0] = e.kappa * (1.0 + w_k)
        grad[1:4] = inst.Phi.T @ weighted[1:4]
        grad[1] += e.sigma_forc / e.M_pre * weighted[4]
        grad[4:6] = e.sigma.T @ weighted[4:6]
        grad[4] -= inst.xi_0 * (1.0 + w_k)
        return StageSolution(
            x=float(self.x.value),
            mu=float(np.clip(self.mu.value, 0.0, 1.0)),
            value=float(self.problem.value + g0),
            gradient=grad,
            slopes_next=weighted / inst.beta,
            status=self.problem.status,
        )
