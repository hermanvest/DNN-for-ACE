import tensorflow as tf
import numpy as np

from typing import Dict, Any
from environments.deqn_ace_dice.equations_of_motion.eom_base import Eom_Base
from utils.debug import assert_valid
from environments.deqn_ace_dice.computation_utils import custom_sigmoid


class Loss_Ace_Dice:
    def __init__(
        self,
        parameters_config: Dict[str, Any],
        equations_of_motion: Eom_Base,
    ) -> None:
        # Variable initialization based on the config files parameters
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                tensor_value = tf.constant(variable_value, dtype=tf.float32)
                setattr(self, variable_name, tensor_value)

        self.beta = tf.constant(
            (1 / (1 + self.prtp)) ** self.timestep, dtype=tf.float32
        )
        self.equations_of_motion = equations_of_motion
        # Should place the create_sigma_transitions in a utility file.
        self.sigma_transition = self.equations_of_motion.create_sigma_transitions()
        self.Phi = self.equations_of_motion.create_Phi_transitions()

    ################ HELPER FUNCITONS ################
    def fischer_burmeister_function(self, a: float, b: float):
        powers = tf.pow(a, 2) + tf.pow(b, 2)
        square_roots = tf.sqrt(powers)

        return a + b - square_roots

    ################ PENALTY BOUNDS ################
    # Penalizing for negative consumption and so on...
    def penalty_bounds_of_policy(self):
        raise NotImplementedError

    ################ INDIVIDUAL LOSS FUNCTIONS ################
    def ell_1(
        self, x_t: tf.Tensor, lambda_k_t: tf.Tensor, lambda_k_tplus: tf.Tensor
    ) -> tf.Tensor:
        """Loss function based on FOC for x_t. The function assumes that appropriate bounds on x_t are applied.

        Args:
            x_t (tf.Tensor): consumption rate in time t
            lambda_k_t (tf.Tensor): shadow value of capital in t
            lambda_k_tplus (tf.Tensor): shadow value of capital in t+1

        Returns:
            loss (tf.Tensor)
        """
        parenthesis = (
            lambda_k_t + tf.pow(self.beta, self.timestep) * lambda_k_tplus * self.kappa
        )

        return (1 / x_t) - (1 / (1 - x_t)) * parenthesis

    def ell_2(
        self,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        t: int,
        lambda_k_t: tf.Tensor,
        lambda_k_tplus: tf.Tensor,
        lambda_m_1_t: tf.Tensor,
        lambda_m_tplus: tf.Tensor,
        lambda_tau_1_t: tf.Tensor,
        lambda_E_t: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on FOC for E_t. The funciton assumes that appropriate bounds on E_t are applied.

        Args:
            E_t (tf.Tensor): emissions level at time t
            k_t (tf.Tensor): capital stock at time t
            t (int): time period
            lambda_k_t (tf.Tensor): Multiplier for capital stock at time t
            lambda_k_tplus (tf.Tensor): Multiplier for capital stock at time t+1
            lambda_m_1_t (tf.Tensor): Multiplier for environmental variable M at time t
            lambda_m_tplus (tf.Tensor): Vector of multipliers for environmental variable M at time t+1
            lambda_tau_1_t (tf.Tensor): Multiplier for tau at time t

        Returns:
            tf.Tensor: loss
        """
        theta_1_t = self.equations_of_motion.theta_1[t]
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        mu_t = 1 - E_t / E_t_BAU

        # Calculaing dlogF/dE_t
        numerator = theta_1_t * self.theta_2 * tf.pow(mu_t, self.theta_2 - 1)
        denominator = E_t_BAU * (1 - theta_1_t * tf.pow(mu_t, self.theta_2))
        d_logF_d_E_t = numerator / denominator

        # Calculating dV_t+1/dE_t
        dm_tplus_part = (
            tf.tensordot(
                lambda_m_tplus, self.Phi[:, 0], axes=1
            )  # Dotprod same as multiplying 1x3 with 3x1
            + (lambda_tau_1_t * self.sigma_forc) / self.M_pre
        )
        dk_tplus_part = lambda_k_tplus * self.kappa * d_logF_d_E_t
        d_V_tplus_d_E_t = dm_tplus_part + dk_tplus_part

        return (
            d_logF_d_E_t * (1 + lambda_k_t)
            + lambda_m_1_t
            + lambda_E_t
            + tf.pow(self.beta, self.timestep) * d_V_tplus_d_E_t
        )

    def ell_3(self, lambda_E_t: tf.Tensor, E_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on Fischer-Burmeister function for E_t \geq 0.

        Args:
            lambda_E_t (tf.Tensor): lagrange multiplier for the constraint on E_t \geq 0
            E_t (tf.Tensor): emissions at time t

        Returns:
            loss (tf.Tensor)
        """
        fb = self.fischer_burmeister_function(lambda_E_t, E_t)
        return fb

    def ell_4(
        self, lambda_t_BAU: tf.Tensor, E_t: tf.Tensor, k_t: tf.Tensor, t: tf.Tensor
    ) -> tf.Tensor:
        """Loss function based on Fischer-Burmeister function for E_t \leq E_t_BAU.

        Args:
            lambda_t_BAU (flaot): lagrange multiplier for the condition that E_t \leq E_t_BAU
            E_t (tf.Tensor): emissions at time t
            k_t (tf.Tensor): log capital at time t
            t (tf.Tensor): timestep

        Returns:
            loss (tf.Tensor)
        """
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        emissions_diff = E_t_BAU - E_t
        return self.fischer_burmeister_function(lambda_t_BAU, emissions_diff)

    def ell_5_7(
        self,
        lambda_m_t: tf.Tensor,
        lambda_m_tplus: tf.Tensor,
        lambda_tau_1_tplus: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on FOC for M_{t+1}.

        Args:
            lambda_m_t (tf.Tensor): vector of langrange multipliers for the carbon stocks at t
            lambda_m_tplus (tf.Tensor): vector of langrange multipliers for the carbon stocks at t+1
            lambda_tau_1_tplus (tf.Tensor): lagrange multiplier for transformed temperature 1 at time t+1

        Returns:
            loss (tf.Tensor)
        """
        # Ensuring that the lambda vector is of the correct shape
        lambda_m_t_reshaped = tf.reshape(lambda_m_t, (3, 1))
        lambda_m_tplus_reshaped = tf.reshape(lambda_m_tplus, (3, 1))
        phi_transposed = tf.transpose(self.Phi)
        transitions = tf.matmul(phi_transposed, lambda_m_tplus_reshaped)

        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])
        forc = lambda_tau_1_tplus * self.sigma_forc * (1 / self.M_pre)

        loss = (
            tf.pow(self.beta, self.timestep) * (transitions + e_1 * forc)
            - lambda_m_t_reshaped
        )

        return tf.reduce_sum(loss)

    def ell_8_9(
        self,
        lambda_tau_t: tf.Tensor,
        lambda_tau_tplus: tf.Tensor,
        lambda_k_tplus: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on FOC for tau_{t+1}.

        Args:
            lambda_tau_t (tf.Tensor): vector of lagrange multipliers for transformed temperatures at time t
            lambda_tau_tplus (tf.Tensor): vector of lagrange multipliers for transformed temperatures at time t+1
            lambda_k_tplus: (tf.Tensor): lagrange multiplier for capital equation of motion in t+1

        Returns:
            loss (tf.Tensor)
        """
        # Ensuring that lambda_tau is of correct shape
        lambda_tau_t_reshaped = tf.reshape(lambda_tau_t, (2, 1))
        lambda_tau_tplus_reshaped = tf.reshape(lambda_tau_tplus, (2, 1))

        # Calculating transitions
        sigma_transposed = tf.transpose(self.sigma_transition)
        transitions = tf.matmul(sigma_transposed, lambda_tau_tplus_reshaped)

        # Calculating Forcing
        e_1 = tf.constant([1, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[2, 1])
        forc = e_1 * self.xi_0 * (1 + lambda_k_tplus)

        loss = (
            tf.pow(self.beta, self.timestep) * (transitions - forc)
            - lambda_tau_t_reshaped
        )
        return tf.reduce_sum(loss)

    def ell_10(
        self,
        v_t: tf.Tensor,
        v_tplus: tf.Tensor,
        x_t: tf.Tensor,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        tau_1_t: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on Bellman Equation. The function assumes that appropriate bounds on x_t are applied.

        Args:
            v_t (tf.Tensor): value function of state at time t
            v_tplus (tf.Tensor): value function of state at time t+1
            x_t (tf.Tensor): consumption rate
            E_t (tf.Tensor): emissions at time t
            k_t (tf.Tensor): log capital at time t
            tau_1_t (tf.Tensor): transformed temperature 1
            t (tf.Tensor): timestep

        Returns:
            loss (tf.Tensor)
        """
        production = self.equations_of_motion.log_Y_t(k_t, E_t, t)
        damages = -self.xi_0 * tau_1_t + self.xi_0

        return (
            tf.math.log(x_t)
            + production
            + damages
            + tf.pow(self.beta, self.timestep) * v_tplus
            - v_t
        )

    ################ MAIN LOSS FUNCTION CALLED FROM ENV ################
    def squared_error_for_transition(
        self, s_t: tf.Tensor, a_t: tf.Tensor, s_tplus: tf.Tensor, a_tplus: tf.Tensor
    ) -> float:
        """Returns the squared error for the state transition from s_t to s_tplus, taking action a_t in the environment

        Args:
            s_t (tf.Tensor): state variables at time t
            a_t (tf.Tensor): action variables at time t
            s_tplus (tf.Tensor): state variables at time t+1
            a_tplus (tf.Tensor): action variables at time t+1

        Returns:
            float: squared error without penalty
        """
        assert_valid(s_t, "s_t")
        assert_valid(s_tplus, "s_tplus")
        assert_valid(a_t, "a_t")
        assert_valid(a_tplus, "a_tplus")

        # action variables t
        x_t = a_t[0]
        E_t = a_t[1]

        v_t = a_t[2]
        lambda_k_t = a_t[3]
        lambda_m_t = a_t[4:7]  # Indexes look wierd, but are correct
        lambda_m_1_t = lambda_m_t[0]
        lambda_tau_t = a_t[7:9]  # Indexes look wierd, but are correct
        lambda_tau_1_t = lambda_tau_t[0]
        lambda_t_BAU = a_t[9]
        lambda_E_t = a_t[10]

        # action variables t+1
        v_tplus = a_tplus[2]
        lambda_k_tplus = a_tplus[3]
        lambda_m_tplus = a_tplus[4:7]  # Indexes look wierd, but are correct
        lambda_tau_tplus = a_tplus[7:9]  # Indexes look wierd, but are correct
        lambda_tau_1_tplus = lambda_tau_tplus[0]

        # state variables t
        k_t = s_t[0]
        tau_t = s_t[4:6]  # Indexes look wierd, but are correct
        tau_1_t = tau_t[0]
        t = tf.cast(s_t[6], tf.int32)

        # Adjustment of E_t:
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        E_t = custom_sigmoid(E_t, E_t_BAU)

        # TODO: This is an abomination. Need to find time to make this prettier.
        loss_functions = [
            ((self.ell_1), (x_t, lambda_k_t, lambda_k_tplus)),
            (
                (self.ell_2),
                (
                    E_t,
                    k_t,
                    t,
                    lambda_k_t,
                    lambda_k_tplus,
                    lambda_m_1_t,
                    lambda_m_tplus,
                    lambda_tau_1_t,
                    lambda_E_t,
                ),
            ),
            ((self.ell_3), (lambda_E_t, E_t)),
            ((self.ell_4), (lambda_t_BAU, E_t, k_t, t)),
            ((self.ell_5_7), (lambda_m_t, lambda_m_tplus, lambda_tau_1_tplus)),
            ((self.ell_8_9), (lambda_tau_t, lambda_tau_tplus, lambda_k_tplus)),
            ((self.ell_10), (v_t, v_tplus, x_t, E_t, k_t, tau_1_t, t)),
        ]

        total_loss = tf.constant(0.0, dtype=tf.float32)
        for func, args in loss_functions:
            total_loss += tf.square(func(*args))

        return total_loss
