"""ACE with DICE functional forms, vectorized over a batch of states.

State vector s_t (7):  [k_t, M_1, M_2, M_3, tau_1, tau_2, t]
    k_t is log capital, M in GtC, tau are transformed temperatures, t the period.

Raw network output (11): [x, E, V, lambda_k, lambda_M (3), lambda_tau (2),
    lambda_BAU, lambda_E]
    `decode_actions` maps raw outputs onto economic quantities. This is the only
    place where emissions are bounded to (0, E_BAU), so every equation of
    motion and residual works with the same emission level.
"""

from typing import Dict

import numpy as np
import tensorflow as tf

from ace_deqn.exogenous import build_exogenous_paths

# Molar mass ratio CO2 / C, used to convert GtCO2 into GtC
CO2_PER_C = 44.0095 / 12.011

STATE_NAMES = ["k", "M_1", "M_2", "M_3", "tau_1", "tau_2", "t"]
ACTION_NAMES = [
    "x",
    "E",
    "V",
    "lambda_k",
    "lambda_M_1",
    "lambda_M_2",
    "lambda_M_3",
    "lambda_tau_1",
    "lambda_tau_2",
    "lambda_BAU",
    "lambda_E",
]
EPSILON = 1e-6


def safe_sigmoid(x: tf.Tensor) -> tf.Tensor:
    return EPSILON + (1 - 2 * EPSILON) * tf.sigmoid(x)


class AceDice:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.version = config["version"]
        economy = config["economy"]
        climate = config["climate"]

        self.t_max = config["simulation"]["t_max"]
        self.n_periods = self.t_max + config["simulation"]["t_extra"] + 2

        # Parameters as plain floats for numpy and tf alike
        self.kappa = float(economy["kappa"])
        self.timestep = float(economy["timestep"])
        self.beta = float((1 / (1 + economy["prtp"])) ** economy["timestep"])
        self.theta_2 = float(economy["theta_2"])
        self.xi_0 = float(climate["xi_0"])
        self.xi_1 = float(climate["xi_1"])
        self.M_pre = float(climate["M_pre"])
        g_k, delta_k = economy["g_k"], economy["delta_k"]
        self.log_capital_factor = float(
            np.log((1 + g_k * self.timestep) / (delta_k + g_k))
        )

        # Emissions in GtCO2 per year raise the carbon stock in GtC over a period
        self.zeta = self.timestep / CO2_PER_C

        # Transition matrices
        xi = np.array(climate["Xi"]) * self.timestep / 5.0
        self.Phi_np = np.array(
            [
                [1.0 - xi[0], xi[2], 0.0],
                [xi[0], 1.0 - xi[1] - xi[2], xi[3]],
                [0.0, xi[1], 1.0 - xi[3]],
            ]
        )
        s_up_1, s_down_1 = climate["sigma_up_1"], climate["sigma_down_1"]
        s_up_2 = climate["sigma_up_2"]
        self.sigma_np = np.array(
            [[1.0 - s_up_1 - s_down_1, s_down_1], [s_up_2, 1.0 - s_up_2]]
        )
        self.sigma_forc = float(1.0 - self.sigma_np[0].sum())

        self.Phi = tf.constant(self.Phi_np, tf.float32)
        self.sigma = tf.constant(self.sigma_np, tf.float32)

        # Exogenous paths
        self.exo = build_exogenous_paths(self.version, economy, self.n_periods)
        self.A = tf.constant(self.exo.A, tf.float32)
        self.N = tf.constant(self.exo.N, tf.float32)
        self.carbon_intensity = tf.constant(self.exo.sigma, tf.float32)
        self.theta_1 = tf.constant(self.exo.theta_1, tf.float32)
        self.E_land = tf.constant(self.exo.E_land, tf.float32)

        scales = config["scales"]
        self.scale = {k: float(v) for k, v in scales.items()}
        self.output_scale = tf.constant(
            [
                1.0,
                1.0,
                scales["V"],
                scales["lambda_k"],
                scales["lambda_M"],
                scales["lambda_M"],
                scales["lambda_M"],
                scales["lambda_tau"],
                scales["lambda_tau"],
                scales["lambda_kkt"],
                scales["lambda_kkt"],
            ],
            tf.float32,
        )

    ################ STATES ################
    def initial_state(self, batch: int = 1) -> tf.Tensor:
        init = self.config["initial_state"]
        s0 = np.concatenate(
            [[np.log(init["K_0"])], init["M_0"], init["tau_0"], [0.0]]
        ).astype(np.float32)
        return tf.tile(tf.constant(s0)[None, :], [batch, 1])

    @staticmethod
    def preprocess(s: tf.Tensor, varsigma: float = 0.02) -> tf.Tensor:
        """Network input: log capital, carbon in 1000 GtC, tau, scaled time."""
        t_hat = 1.0 - tf.exp(-varsigma * s[:, 6:7])
        return tf.concat([s[:, 0:1], s[:, 1:4] / 1000.0, s[:, 4:6], t_hat], axis=1)

    def _period(self, s: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.round(s[:, 6]), tf.int32)

    ################ PRODUCTION ################
    def log_Y_gross(self, s: tf.Tensor) -> tf.Tensor:
        t = self._period(s)
        A = tf.gather(self.A, t)
        N = tf.gather(self.N, t)
        return tf.math.log(A) + (1 - self.kappa) * tf.math.log(N / 1000.0) + self.kappa * s[:, 0]

    def E_BAU(self, s: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.carbon_intensity, self._period(s)) * tf.exp(self.log_Y_gross(s))

    def log_F(self, s: tf.Tensor, E: tf.Tensor) -> tf.Tensor:
        """Log output after abatement costs, log F_t(.) in the thesis."""
        theta_1 = tf.gather(self.theta_1, self._period(s))
        mu = 1.0 - E / self.E_BAU(s)
        return self.log_Y_gross(s) + tf.math.log(1.0 - theta_1 * tf.pow(mu, self.theta_2))

    def dlogF_dE(self, s: tf.Tensor, E: tf.Tensor) -> tf.Tensor:
        """Derivative of log F w.r.t. emissions, with E_BAU taken as given."""
        theta_1 = tf.gather(self.theta_1, self._period(s))
        E_BAU = self.E_BAU(s)
        mu = 1.0 - E / E_BAU
        return (
            theta_1 * self.theta_2 * tf.pow(mu, self.theta_2 - 1)
            / (E_BAU * (1.0 - theta_1 * tf.pow(mu, self.theta_2)))
        )

    def dlogF_dk(self) -> float:
        """Derivative of log F w.r.t. log capital, with E_BAU taken as given."""
        return self.kappa

    def log_utility(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        return (
            tf.math.log(a["x"]) + self.log_F(s, a["E"])
            - self.xi_0 * s[:, 4] + self.xi_0
        )

    ################ ACTIONS ################
    def decode_actions(self, s: tf.Tensor, raw: tf.Tensor) -> Dict[str, tf.Tensor]:
        scaled = raw * self.output_scale
        return {
            "x": safe_sigmoid(raw[:, 0]),
            "E": self.E_BAU(s) * safe_sigmoid(raw[:, 1]),
            "V": scaled[:, 2],
            "lambda_k": scaled[:, 3],
            "lambda_M": scaled[:, 4:7],
            "lambda_tau": scaled[:, 7:9],
            "lambda_BAU": scaled[:, 9],
            "lambda_E": scaled[:, 10],
        }

    ################ EQUATIONS OF MOTION ################
    def step(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        t = self._period(s)
        k_next = (
            self.log_F(s, a["E"]) - self.xi_0 * s[:, 4] + self.xi_0
            + tf.math.log(1.0 - a["x"]) + self.log_capital_factor
        )
        emissions = a["E"] + tf.gather(self.E_land, t)
        M_next = tf.matmul(s[:, 1:4], self.Phi, transpose_b=True)
        M_next += self.zeta * emissions[:, None] * tf.constant([[1.0, 0.0, 0.0]])
        tau_next = tf.matmul(s[:, 4:6], self.sigma, transpose_b=True)
        tau_next += (self.sigma_forc * s[:, 1:2] / self.M_pre) * tf.constant([[1.0, 0.0]])
        return tf.concat([k_next[:, None], M_next, tau_next, s[:, 6:7] + 1.0], axis=1)

    ################ DIAGNOSTICS ################
    def temperature(self, tau: np.ndarray) -> np.ndarray:
        return np.log(tau) / self.xi_1

    def net_output(self, s: tf.Tensor, E: tf.Tensor) -> tf.Tensor:
        return tf.exp(self.log_F(s, E) - self.xi_0 * s[:, 4] + self.xi_0)

    def social_cost_of_carbon(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        """SCC in USD per tCO2 implied by the costates.

        From the emissions FOC, the marginal log-output value of a GtCO2 per year
        equals -zeta * lambda_M1 / (1 + lambda_k). Multiplying by net output
        (trillion USD) and 1000 converts it to USD per tCO2.
        """
        price = -self.zeta * a["lambda_M"][:, 0] / (1.0 + a["lambda_k"])
        return 1000.0 * self.net_output(s, a["E"]) * price
