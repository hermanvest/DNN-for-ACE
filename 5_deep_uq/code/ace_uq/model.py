"""ACE with DICE functional forms and uncertain parameters as pseudo-states.

State vector s_t (10):  [k_t, M_1, M_2, M_3, tau_1, tau_2, t, prtp, xi_0, carbon_exchange]
    The first seven entries are the economic states of part 2: log capital, carbon
    in GtC, transformed temperatures and the period. The last three are the
    uncertain parameters. They are constant along a trajectory, so the equations
    of motion copy them forward, and they are network inputs like any other state.
    One network therefore solves the model for every parameter value at once
    (Friedl, Kübler, Scheidegger and Usui, Deep Uncertainty Quantification).

Raw network output (11): [x, E, V, lambda_k, lambda_M (3), lambda_tau (2),
    lambda_BAU, lambda_E], decoded by `decode_actions` as in part 2.
"""

from typing import Dict

import numpy as np
import tensorflow as tf

from ace_uq.exogenous import build_exogenous_paths

# Molar mass ratio CO2 / C, used to convert GtCO2 into GtC
CO2_PER_C = 44.0095 / 12.011

ECONOMIC_STATES = ["k", "M_1", "M_2", "M_3", "tau_1", "tau_2", "t"]
PARAMETERS = ["prtp", "xi_0", "carbon_exchange"]
STATE_NAMES = ECONOMIC_STATES + PARAMETERS
N_ECONOMIC = len(ECONOMIC_STATES)
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


class AceDiceUQ:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.version = config["version"]
        economy = config["economy"]
        climate = config["climate"]

        self.t_max = config["simulation"]["t_max"]
        self.n_periods = self.t_max + config["simulation"]["t_extra"] + 2

        # Certain parameters
        self.kappa = float(economy["kappa"])
        self.timestep = float(economy["timestep"])
        self.theta_2 = float(economy["theta_2"])
        self.xi_1 = float(climate["xi_1"])
        self.M_pre = float(climate["M_pre"])
        g_k, delta_k = economy["g_k"], economy["delta_k"]
        self.log_capital_factor = float(np.log((1 + g_k * self.timestep) / (delta_k + g_k)))
        self.zeta = self.timestep / CO2_PER_C

        # Carbon cycle: atmosphere <-> upper ocean exchange (xi_0 and xi_2 of DICE)
        # is scaled by the uncertain multiplier, which keeps every column summing to one
        self.Xi = np.array(climate["Xi"]) * self.timestep / 5.0
        s_up_1, s_down_1 = climate["sigma_up_1"], climate["sigma_down_1"]
        s_up_2 = climate["sigma_up_2"]
        self.sigma_np = np.array([[1.0 - s_up_1 - s_down_1, s_down_1], [s_up_2, 1.0 - s_up_2]])
        self.sigma_forc = float(1.0 - self.sigma_np[0].sum())
        self.sigma = tf.constant(self.sigma_np, tf.float32)

        # Baseline values of the uncertain parameters and their training box
        self.baseline = np.array([economy["prtp"], climate["xi_0"], 1.0])
        box = config["uncertain_parameters"]
        self.low = np.array([box[p]["low"] for p in PARAMETERS])
        self.high = np.array([box[p]["high"] for p in PARAMETERS])

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
            [1.0, 1.0, scales["V"], scales["lambda_k"]]
            + [scales["lambda_M"]] * 3
            + [scales["lambda_tau"]] * 2
            + [scales["lambda_kkt"]] * 2,
            tf.float32,
        )

    ################ PARAMETERS ################
    def beta(self, s: tf.Tensor) -> tf.Tensor:
        return tf.pow(1.0 + s[:, 7], -self.timestep)

    def xi_0(self, s: tf.Tensor) -> tf.Tensor:
        return s[:, 8]

    def Phi_np(self, carbon_exchange: float) -> np.ndarray:
        xi = self.Xi.copy()
        xi[[0, 2]] *= carbon_exchange
        return np.array(
            [
                [1.0 - xi[0], xi[2], 0.0],
                [xi[0], 1.0 - xi[1] - xi[2], xi[3]],
                [0.0, xi[1], 1.0 - xi[3]],
            ]
        )

    def Phi(self, s: tf.Tensor) -> tf.Tensor:
        """Carbon transition matrix of every sample, shape [batch, 3, 3]."""
        base = tf.constant(self.Phi_np(0.0), tf.float32)
        exchange = tf.constant(self.Phi_np(1.0) - self.Phi_np(0.0), tf.float32)
        return base[None] + s[:, 9, None, None] * exchange[None]

    def sample_parameters(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform draws from the training box."""
        return self.low + (self.high - self.low) * rng.random((n, len(PARAMETERS)))

    ################ STATES ################
    def initial_state(self, parameters: np.ndarray) -> tf.Tensor:
        """Initial states for a batch of parameter vectors, shape [batch, 10]."""
        parameters = np.atleast_2d(parameters)
        init = self.config["initial_state"]
        s0 = np.concatenate([[np.log(init["K_0"])], init["M_0"], init["tau_0"], [0.0]])
        s0 = np.tile(s0, (len(parameters), 1))
        return tf.constant(np.concatenate([s0, parameters], axis=1), tf.float32)

    def preprocess(self, s: tf.Tensor, varsigma: float = 0.02) -> tf.Tensor:
        """Network input: the part 2 inputs, plus the parameters mapped onto [-1, 1]."""
        t_hat = 1.0 - tf.exp(-varsigma * s[:, 6:7])
        low = tf.constant(self.low, tf.float32)
        high = tf.constant(self.high, tf.float32)
        p_hat = 2.0 * (s[:, 7:] - low) / (high - low) - 1.0
        return tf.concat([s[:, 0:1], s[:, 1:4] / 1000.0, s[:, 4:6], t_hat, p_hat], axis=1)

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
        """Log output after abatement costs."""
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
        return self.kappa

    def log_net_output(self, s: tf.Tensor, E: tf.Tensor) -> tf.Tensor:
        xi_0 = self.xi_0(s)
        return self.log_F(s, E) - xi_0 * s[:, 4] + xi_0

    def log_utility(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        return tf.math.log(a["x"]) + self.log_net_output(s, a["E"])

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
        k_next = self.log_net_output(s, a["E"]) + tf.math.log(1.0 - a["x"]) + self.log_capital_factor
        emissions = a["E"] + tf.gather(self.E_land, t)
        M_next = tf.einsum("bij,bj->bi", self.Phi(s), s[:, 1:4])
        M_next += self.zeta * emissions[:, None] * tf.constant([[1.0, 0.0, 0.0]])
        tau_next = tf.matmul(s[:, 4:6], self.sigma, transpose_b=True)
        tau_next += (self.sigma_forc * s[:, 1:2] / self.M_pre) * tf.constant([[1.0, 0.0]])
        return tf.concat([k_next[:, None], M_next, tau_next, s[:, 6:7] + 1.0, s[:, 7:]], axis=1)

    ################ DIAGNOSTICS ################
    def temperature(self, tau: np.ndarray) -> np.ndarray:
        return np.log(tau) / self.xi_1

    def net_output(self, s: tf.Tensor, E: tf.Tensor) -> tf.Tensor:
        return tf.exp(self.log_net_output(s, E))

    def social_cost_of_carbon(self, s: tf.Tensor, a: Dict[str, tf.Tensor]) -> tf.Tensor:
        """SCC in USD per tCO2 implied by the costates (see part 2)."""
        price = -self.zeta * a["lambda_M"][:, 0] / (1.0 + a["lambda_k"])
        return 1000.0 * self.net_output(s, a["E"]) * price
