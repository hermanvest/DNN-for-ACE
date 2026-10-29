"""ACE with DICE functional forms as a differentiable environment.

State s_t (7): [k_t, M_1, M_2, M_3, tau_1, tau_2, t]
    k_t is log capital, M in GtC, tau are transformed temperatures, t the period.
Action (2): consumption rate x_t in (0, 1) and emissions as a share of
    business-as-usual emissions, so the abatement rate is mu_t = 1 - share.

Business-as-usual emissions follow an exogenous reference path: sigma_t times
gross output along the analytic optimal path. On that path they coincide with
the DICE definition E_BAU = sigma_t Y_gross(K_t); off the path they stay
exogenous, which is the assumption under which ACE has a closed-form solution.
"""

from typing import Dict

import numpy as np
import tensorflow as tf

from ace_grpo.benchmark import analytic_costates, optimal_abatement
from ace_grpo.exogenous import build_exogenous_paths

CO2_PER_C = 44.0095 / 12.011
EPSILON = 1e-6


class AceEnv:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.version = config["version"]
        economy, climate = config["economy"], config["climate"]

        self.t_max = config["simulation"]["t_max"]
        self.n_periods = self.t_max + config["simulation"]["t_extra"] + 2

        self.kappa = float(economy["kappa"])
        self.timestep = float(economy["timestep"])
        self.beta = float((1 / (1 + economy["prtp"])) ** economy["timestep"])
        self.theta_2 = float(economy["theta_2"])
        self.xi_0 = float(climate["xi_0"])
        self.xi_1 = float(climate["xi_1"])
        self.M_pre = float(climate["M_pre"])
        g_k, delta_k = economy["g_k"], economy["delta_k"]
        self.log_capital_factor = float(np.log((1 + g_k * self.timestep) / (delta_k + g_k)))
        self.zeta = self.timestep / CO2_PER_C

        xi = np.array(climate["Xi"]) * self.timestep / 5.0
        self.Phi_np = np.array(
            [
                [1.0 - xi[0], xi[2], 0.0],
                [xi[0], 1.0 - xi[1] - xi[2], xi[3]],
                [0.0, xi[1], 1.0 - xi[3]],
            ]
        )
        s_up_1, s_down_1, s_up_2 = climate["sigma_up_1"], climate["sigma_down_1"], climate["sigma_up_2"]
        self.sigma_np = np.array([[1.0 - s_up_1 - s_down_1, s_down_1], [s_up_2, 1.0 - s_up_2]])
        self.sigma_forc = float(1.0 - self.sigma_np[0].sum())
        self.Phi = tf.constant(self.Phi_np, tf.float32)
        self.sigma = tf.constant(self.sigma_np, tf.float32)

        self.exo = build_exogenous_paths(self.version, economy, self.n_periods)
        self.A = tf.constant(self.exo.A, tf.float32)
        self.N = tf.constant(self.exo.N, tf.float32)
        self.theta_1 = tf.constant(self.exo.theta_1, tf.float32)
        self.E_land = tf.constant(self.exo.E_land, tf.float32)

        self.costates = analytic_costates(
            self.beta, self.kappa, self.xi_0, self.sigma_np, self.Phi_np,
            self.sigma_forc, self.M_pre, self.zeta,
        )
        self.E_BAU_path = tf.constant(self._reference_E_BAU(), tf.float32)
        self.mu_star = optimal_abatement(
            self.exo.theta_1, self.E_BAU_path.numpy(), self.theta_2, self.costates.carbon_price
        )

    ################ STATES ################
    def initial_state(self, batch: int = 1) -> tf.Tensor:
        init = self.config["initial_state"]
        s0 = np.concatenate([[np.log(init["K_0"])], init["M_0"], init["tau_0"], [0.0]])
        return tf.tile(tf.constant(s0, tf.float32)[None, :], [batch, 1])

    @staticmethod
    def preprocess(s: tf.Tensor, varsigma: float = 0.02) -> tf.Tensor:
        """Network input: log capital, carbon in 1000 GtC, tau, scaled time."""
        t_hat = 1.0 - tf.exp(-varsigma * s[:, 6:7])
        return tf.concat([s[:, 0:1], s[:, 1:4] / 1000.0, s[:, 4:6], t_hat], axis=1)

    def _period(self, s: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.round(s[:, 6]), tf.int32)

    ################ ECONOMY ################
    def log_Y_gross(self, s: tf.Tensor) -> tf.Tensor:
        t = self._period(s)
        return (
            tf.math.log(tf.gather(self.A, t))
            + (1 - self.kappa) * tf.math.log(tf.gather(self.N, t) / 1000.0)
            + self.kappa * s[:, 0]
        )

    def E_BAU(self, s: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.E_BAU_path, self._period(s))

    def log_F(self, s: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
        theta_1 = tf.gather(self.theta_1, self._period(s))
        return self.log_Y_gross(s) + tf.math.log(1.0 - theta_1 * tf.pow(mu, self.theta_2))

    def log_net_output(self, s: tf.Tensor, mu: tf.Tensor) -> tf.Tensor:
        return self.log_F(s, mu) - self.xi_0 * s[:, 4] + self.xi_0

    def utility(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Per-period reward: log consumption."""
        return tf.math.log(a["x"]) + self.log_net_output(s, a["mu"])

    def step(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        t = self._period(s)
        k_next = (
            self.log_net_output(s, a["mu"]) + tf.math.log(1.0 - a["x"]) + self.log_capital_factor
        )
        emissions = (1.0 - a["mu"]) * self.E_BAU(s) + tf.gather(self.E_land, t)
        M_next = tf.matmul(s[:, 1:4], self.Phi, transpose_b=True)
        M_next += self.zeta * emissions[:, None] * tf.constant([[1.0, 0.0, 0.0]])
        tau_next = tf.matmul(s[:, 4:6], self.sigma, transpose_b=True)
        tau_next += (self.sigma_forc * s[:, 1:2] / self.M_pre) * tf.constant([[1.0, 0.0]])
        return tf.concat([k_next[:, None], M_next, tau_next, s[:, 6:7] + 1.0], axis=1)

    ################ ANALYTIC POLICY ################
    def analytic_action(self, s: tf.Tensor) -> Dict[str, tf.Tensor]:
        """The optimal policy: constant consumption rate, abatement depending only on t."""
        batch = tf.shape(s)[0]
        mu = tf.gather(tf.constant(np.clip(self.mu_star, EPSILON, 1 - EPSILON), tf.float32), self._period(s))
        return {"x": tf.fill([batch], tf.constant(self.costates.x, tf.float32)), "mu": mu}

    def _reference_E_BAU(self) -> np.ndarray:
        """sigma_t Y_gross(K_t) along the analytic optimal path with endogenous E_BAU."""
        sigma = self.exo.sigma
        c = self.costates
        s = self.initial_state(1)
        E_BAU = np.zeros(self.n_periods)
        for t in range(self.n_periods):
            E_BAU[t] = sigma[t] * float(tf.exp(self.log_Y_gross(s))[0])
            if t == self.n_periods - 1:
                break
            mu = optimal_abatement(self.exo.theta_1[t], E_BAU[t], self.theta_2, c.carbon_price)
            mu = float(np.clip(mu, EPSILON, 1 - EPSILON))
            self.E_BAU_path = tf.constant(E_BAU, tf.float32)
            s = self.step(s, {"x": tf.constant([c.x], tf.float32), "mu": tf.constant([mu], tf.float32)})
        return E_BAU

    def rollout_value(self, s: tf.Tensor, policy, n: int = 250) -> tf.Tensor:
        """Discounted utility of following `policy` for n periods from each state in s."""
        value = tf.zeros(tf.shape(s)[0])
        discount = 1.0
        for _ in range(n):
            a = policy(s)
            value += discount * self.utility(s, a)
            discount *= self.beta
            s = self.step(s, a)
        return value

    ################ DIAGNOSTICS ################
    def temperature(self, tau) -> np.ndarray:
        return np.log(np.asarray(tau)) / self.xi_1

    def scc(self, s: tf.Tensor, mu: tf.Tensor, lambda_k: tf.Tensor, lambda_M1: tf.Tensor) -> tf.Tensor:
        """SCC in USD/tCO2 from the costates: -1000 Y_net zeta lambda_M1 / (1 + lambda_k)."""
        return -1000.0 * tf.exp(self.log_net_output(s, mu)) * self.zeta * lambda_M1 / (1.0 + lambda_k)
