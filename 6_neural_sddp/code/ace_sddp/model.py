"""ACE with DICE functional forms, as a sequence of convex stage problems.

State s_t (6): [k_t, M_1, M_2, M_3, tau_1, tau_2], with k_t log capital, M carbon in GtC
and tau the transformed temperatures. Decisions in each period: the consumption rate x_t
and the abatement rate mu_t. With business-as-usual emissions on an exogenous path,

    reward    u_t = log x_t + g_t,   g_t = log Y_gross,t(k_t) + log(1 - theta_1,t mu_t^theta_2) - xi_0 tau_1,t + xi_0
    capital   k_t+1 = g_t + log(1 - x_t) + log_capital_factor
    carbon    M_t+1 = Phi M_t + zeta ((1 - mu_t) E_BAU,t + E_land,t) e_1
    climate   tau_t+1 = sigma tau_t + sigma_forc M_1,t / M_pre e_1

The reward and next capital are concave in (x, mu), carbon is affine in mu and the
climate does not depend on the decisions at all, so every stage problem is convex.
The value function is affine in the state, V_t(s) = alpha_t + b . s, with constant slopes
b = (lambda_k, lambda_M, lambda_tau) / beta in terms of the costates of parts 2 to 5.
These slopes are exactly what SDDP's cuts estimate.

An `Economy` holds what all parameter draws share (exogenous paths, initial state).
An `Instance` is one draw of the uncertain parameters (prtp, xi_0, carbon_exchange).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ace_sddp.exogenous import build_exogenous_paths

CO2_PER_C = 44.0095 / 12.011
PARAMETERS = ["prtp", "xi_0", "carbon_exchange"]
STATE_NAMES = ["k", "M_1", "M_2", "M_3", "tau_1", "tau_2"]


@dataclass
class ClosedForm:
    x: float
    slopes: np.ndarray  # dV/ds: [b_k, b_M (3), b_tau (2)]
    carbon_price: float  # -zeta beta b_M1 / (1 + beta b_k), log output per GtCO2/yr


class Economy:
    def __init__(self, config: Dict):
        self.config = config
        self.version = config["version"]
        economy, climate = config["economy"], config["climate"]
        self.horizon = int(config["simulation"]["horizon"])
        self.evaluate = int(config["simulation"]["evaluate"])
        self.n_periods = self.horizon + 2

        self.kappa = float(economy["kappa"])
        self.timestep = float(economy["timestep"])
        self.theta_2 = float(economy["theta_2"])
        self.xi_1 = float(climate["xi_1"])
        self.M_pre = float(climate["M_pre"])
        g_k, delta_k = economy["g_k"], economy["delta_k"]
        self.log_capital_factor = float(np.log((1 + g_k * self.timestep) / (delta_k + g_k)))
        self.zeta = self.timestep / CO2_PER_C

        self.Xi = np.array(climate["Xi"]) * self.timestep / 5.0
        s_up_1, s_down_1, s_up_2 = climate["sigma_up_1"], climate["sigma_down_1"], climate["sigma_up_2"]
        self.sigma = np.array([[1.0 - s_up_1 - s_down_1, s_down_1], [s_up_2, 1.0 - s_up_2]])
        self.sigma_forc = float(1.0 - self.sigma[0].sum())

        self.exo = build_exogenous_paths(self.version, economy, self.n_periods)
        self.log_A = np.log(self.exo.A)
        self.log_N = np.log(self.exo.N / 1000.0)
        self.theta_1 = np.asarray(self.exo.theta_1, dtype=float)
        self.E_land = np.asarray(self.exo.E_land, dtype=float)

        init = config["initial_state"]
        self.s0 = np.concatenate([[np.log(init["K_0"])], init["M_0"], init["tau_0"]]).astype(float)

        box = config["uncertain_parameters"]
        self.low = np.array([box[p]["low"] for p in PARAMETERS])
        self.high = np.array([box[p]["high"] for p in PARAMETERS])
        self.baseline = np.array([economy["prtp"], climate["xi_0"], 1.0])

        # Business-as-usual emissions follow an exogenous reference path: carbon intensity
        # times gross output along the baseline optimum (as in parts 3 and 4)
        self.E_BAU = np.zeros(self.n_periods)
        self.E_BAU = self._reference_E_BAU()

    def Phi(self, exchange: float) -> np.ndarray:
        xi = self.Xi.copy()
        xi[[0, 2]] *= exchange
        return np.array([
            [1.0 - xi[0], xi[2], 0.0],
            [xi[0], 1.0 - xi[1] - xi[2], xi[3]],
            [0.0, xi[1], 1.0 - xi[3]],
        ])

    def log_Y_gross(self, t: int, k: float) -> float:
        return self.log_A[t] + (1 - self.kappa) * self.log_N[t] + self.kappa * k

    def _reference_E_BAU(self) -> np.ndarray:
        base = Instance(self, self.baseline)
        cf = base.closed_form()
        s = self.s0.copy()
        E = np.zeros(self.n_periods)
        for t in range(self.n_periods):
            E[t] = self.exo.sigma[t] * np.exp(self.log_Y_gross(t, s[0]))
            self.E_BAU = E  # the partial path is all step() needs up to t
            mu = optimal_abatement(self.theta_1[t], E[t], self.theta_2, cf.carbon_price)
            if t < self.n_periods - 1:
                s = base.step(t, s, cf.x, mu)
        return E


class Instance:
    """One draw of the uncertain parameters."""

    def __init__(self, economy: Economy, parameters):
        self.economy = economy
        self.parameters = np.asarray(parameters, dtype=float)
        prtp, self.xi_0, exchange = self.parameters
        self.beta = float((1.0 + prtp) ** -economy.timestep)
        self.Phi = economy.Phi(exchange)

    ################ DYNAMICS ################
    def g(self, t: int, s: np.ndarray, mu: float, shock: float = 0.0) -> float:
        e = self.economy
        return e.log_Y_gross(t, s[0]) + shock + np.log(1.0 - e.theta_1[t] * mu**e.theta_2) - self.xi_0 * s[4] + self.xi_0

    def utility(self, t: int, s: np.ndarray, x: float, mu: float, shock: float = 0.0) -> float:
        return np.log(x) + self.g(t, s, mu, shock)

    def step(self, t: int, s: np.ndarray, x: float, mu: float, shock: float = 0.0) -> np.ndarray:
        e = self.economy
        k_next = self.g(t, s, mu, shock) + np.log(1.0 - x) + e.log_capital_factor
        M_next = self.Phi @ s[1:4]
        M_next[0] += e.zeta * ((1.0 - mu) * e.E_BAU[t] + e.E_land[t])
        tau_next = e.sigma @ s[4:6]
        tau_next[0] += e.sigma_forc * s[1] / e.M_pre
        return np.concatenate([[k_next], M_next, tau_next])

    def carbon_price(self, slopes_next: np.ndarray) -> float:
        """Shadow price of emissions in log output, from the slopes of V_t+1."""
        b_k, b_M1 = slopes_next[0], slopes_next[1]
        return -self.economy.zeta * self.beta * b_M1 / (1.0 + self.beta * b_k)

    def scc(self, t: int, s: np.ndarray, mu: float, slopes_next: np.ndarray) -> float:
        """Social cost of carbon in USD per tCO2: the carbon price times net output."""
        return 1000.0 * np.exp(self.g(t, s, mu)) * self.carbon_price(slopes_next)

    ################ CLOSED FORM ################
    def closed_form(self) -> ClosedForm:
        e, beta = self.economy, self.beta
        b_k = e.kappa / (1.0 - beta * e.kappa)
        b_tau = -self.xi_0 * (1.0 + beta * b_k) * np.linalg.solve(np.eye(2) - beta * e.sigma.T, [1.0, 0.0])
        b_M = beta * e.sigma_forc / e.M_pre * b_tau[0] * np.linalg.solve(np.eye(3) - beta * self.Phi.T, [1.0, 0.0, 0.0])
        slopes = np.concatenate([[b_k], b_M, b_tau])
        return ClosedForm(x=1.0 - beta * e.kappa, slopes=slopes, carbon_price=self.carbon_price(slopes))

    def closed_form_path(self, n: int = None) -> Dict[str, np.ndarray]:
        """The optimal path from the initial state, over n periods (default: the full horizon)."""
        e = self.economy
        n = n or e.horizon
        cf = self.closed_form()
        s = e.s0.copy()
        out = {k: [] for k in ("states", "x", "mu", "scc", "utility")}
        for t in range(n):
            mu = optimal_abatement(e.theta_1[t], e.E_BAU[t], e.theta_2, cf.carbon_price)
            out["states"].append(s)
            out["x"].append(cf.x)
            out["mu"].append(mu)
            out["scc"].append(self.scc(t, s, mu, cf.slopes))
            out["utility"].append(self.utility(t, s, cf.x, mu))
            s = self.step(t, s, cf.x, mu)
        out = {k: np.array(v) for k, v in out.items()}
        out["value"] = float(np.sum(out["utility"] * self.beta ** np.arange(n)))
        return out


def optimal_abatement(theta_1, E_BAU, theta_2: float, carbon_price: float) -> float:
    """The abatement rate where the marginal product of emissions equals the carbon price (capped at 1)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        marginal = theta_1 * theta_2 * mid ** (theta_2 - 1) / (E_BAU * (1 - theta_1 * mid**theta_2))
        lo, hi = (mid, hi) if marginal < carbon_price else (lo, mid)
    return 0.5 * (lo + hi)
