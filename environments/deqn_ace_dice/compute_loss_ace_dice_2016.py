import tensorflow as tf
import numpy as np

from typing import Tuple, Dict, Any
from environments.deqn_ace_dice.equations_of_motion_ace_dice_2016 import (
    Equations_of_motion_Ace_Dice_2016,
)
from utils.debug import assert_valid


class Computeloss_Ace_Dice_2016:
    def __init__(
        self,
        parameters_config: Dict[str, Any],
        equations_of_motion: Equations_of_motion_Ace_Dice_2016,
    ) -> None:
        # Variable initialization based on the config files parameters
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                setattr(self, variable_name, variable_value)

        self.beta = tf.constant(
            (1 / (1 + self.prtp)) ** self.timestep, dtype=tf.float32
        )

        self.Phi = tf.constant(self.Phi, dtype=tf.float32)
        self.sigma_transition = tf.constant(self.sigma_transition, dtype=tf.float32)
        self.equations_of_motion = equations_of_motion

    ################ HELPER FUNCITONS ################
    # @tf.function add after debug
    def fischer_burmeister_function(self, a: float, b: float):
        powers = tf.pow(a, 2) + tf.pow(b, 2)
        square_roots = tf.sqrt(powers)

        return a + b - square_roots

    ################ PENALTY BOUNDS ################
    # Penalizing for negative consumption and so on...
    def penalty_bounds_of_policy(self):
        raise NotImplementedError

    ################ INDIVIDUAL LOSS FUNCTIONS ################
    # @tf.function add after debug
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
        parenthesis = lambda_k_t + self.beta * lambda_k_tplus * self.kappa

        return (1 / x_t) - (1 / (1 - x_t)) * parenthesis

    def ell_2(
        self,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        t: int,
        lambda_k_t: tf.Tensor,
        lambda_k_tplus: tf.Tensor,
        lambda_M_1_t: tf.Tensor,
        lambda_M_tplus_vector: tf.Tensor,
        lambda_tau_1_t: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on FOC for E_t. The funciton assumes that appropriate bounds on E_t are applied.

        Args:
            E_t (tf.Tensor): emissions level at time t
            k_t (tf.Tensor): capital stock at time t
            t (int): time period
            lambda_k_t (tf.Tensor): Multiplier for capital stock at time t
            lambda_k_tplus (tf.Tensor): Multiplier for capital stock at time t+1
            lambda_M_1_t (tf.Tensor): Multiplier for environmental variable M at time t
            lambda_M_tplus_vector (tf.Tensor): Vector of multipliers for environmental variable M at time t+1
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
                lambda_M_tplus_vector, self.Phi[:, 0]
            )  # Dotprod same as multiplying 1x3 with 3x1
            + (lambda_tau_1_t * self.sigma_forc) / self.M_pre
        )
        dk_tplus_part = lambda_k_tplus * self.kappa * d_logF_d_E_t
        d_V_tplus_d_E_t = dm_tplus_part + dk_tplus_part

        return (
            d_logF_d_E_t * (1 + lambda_k_t) + lambda_M_1_t + self.beta * d_V_tplus_d_E_t
        )

    # @tf.function add after debug
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

    # @tf.function add after debug
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

    # @tf.function add after debug
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

        loss = self.beta * (transitions + e_1 * forc) - lambda_m_t_reshaped

        return tf.reduce_sum(loss)

    # @tf.function add after debug
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
        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])
        forc = e_1 * self.xi_0 * (1 + lambda_k_tplus)

        loss = self.beta * (transitions - forc) - lambda_tau_t_reshaped
        return tf.reduce_sum(loss)

    # @tf.function add after debug
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

        return tf.math.log(x_t) + production + damages + self.beta * v_tplus - v_t

    ################ MAIN LOSS FUNCTION CALLED FROM ENV ################
    # @tf.function add after debug
    def squared_error_for_transition(
        self, s_t: tf.Tensor, a_t: tf.Tensor, s_tplus: tf.Tensor, a_tplus: tf.Tensor
    ) -> float:
        """Returns the squared error for the state transition from s_t to s_tplus, taking action a_t in the environment

        Args:
            s_t (tf.Tensor): state variables at time t
            a_t (tf.Tensor): action variables at time t
            s_tplus (tf.Tensor): state variables at time t+1

        Returns:
            Tuple[float, float]: (squared error without penalty, squared error with penalty)
        """
        assert_valid(s_t, "s_t")
        assert_valid(s_tplus, "s_tplus")
        assert_valid(a_t, "a_t")
        assert_valid(a_tplus, "a_tplus")

        # action variables t
        x_t = a_t[0]  # Adjusted below
        E_t = a_t[1]  # Adjusted below

        value_func_t = a_t[2]
        lambda_k_t = a_t[3]
        lambda_m_t_vector = a_t[4:7]  # Indexes look wierd, but are correct
        lambda_tau_t_vector = a_t[7:9]  # Indexes look wierd, but are correct
        lambda_t_BAU = a_t[9]
        lambda_E_t = a_t[10]

        # action variables t+1
        value_func_tplus = a_tplus[2]
        lambda_k_tplus = a_tplus[3]
        lambda_m_tplus_vector = a_tplus[4:7]  # Indexes look wierd, but are correct
        lambda_tau_tplus_vector = a_tplus[7:9]  # Indexes look wierd, but are correct

        # state variables t
        k_t = s_t[0]
        tau_t_vector = s_t[4:6]  # Indexes look wierd, but are correct
        t = tf.cast(s_t[6], tf.int32)

        # state variables t+1
        k_tplus = s_tplus[0]

        ## Adjustments
        x_t_adj = tf.raw_ops.ClipByValue(
            x_t, clip_value_min=1e-7, clip_value_max=1 - 1e-7, name=None
        )
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        E_t_adj = tf.minimum(E_t, E_t_BAU)

        # TODO: This is an abomination. Need to find time to make this prettier.
        loss1 = tf.square(self.ell_1(lambda_k_t, lambda_k_tplus))
        loss2_4 = tf.square(
            self.ell_2_4(
                lambda_m_t_vector, lambda_m_tplus_vector, lambda_tau_tplus_vector[0]
            )
        )
        loss5_6 = tf.square(self.ell_5_6(lambda_tau_t_vector, lambda_tau_tplus_vector))
        loss7 = tf.square(
            self.ell_7(k_tplus, x_t_adj, E_t_adj, k_t, tau_t_vector[0], t)
        )
        loss8 = tf.square(self.ell_8(lambda_E_t, E_t_adj))
        loss9 = tf.square(self.ell_9(lambda_t_BAU, E_t_adj, k_t, t))
        loss10 = tf.square(
            self.ell_10(
                value_func_t,
                value_func_tplus,
                x_t_adj,
                E_t_adj,
                k_t,
                tau_t_vector[0],
                t,
            )
        )

        total_loss = loss1 + loss2_4 + loss5_6 + loss7 + loss8 + loss9 + loss10
        total_loss_float32 = tf.cast(total_loss, dtype=tf.float32)

        return total_loss_float32
