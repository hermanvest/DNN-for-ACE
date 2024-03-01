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
    def ell_1(self, lambda_k_t: tf.Tensor, lambda_k_tplus: tf.Tensor) -> tf.Tensor:
        """Loss function based on FOC for k_{t+1}.

        Args:
            lambda_k_t (tf.Tensor): shadow value of capital in t
            lambda_k_t (tf.Tensor): shadow value of capital in t+1

        Returns:
            loss (tf.Tensor)
        """
        return self.beta * lambda_k_tplus * self.kappa - lambda_k_t

    # @tf.function add after debug
    def ell_2_4(
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

        transitions = tf.matmul(self.Phi, lambda_m_tplus_reshaped)

        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])

        forc = lambda_tau_1_tplus * self.sigma_forc * (1 / self.M_pre)

        loss = self.beta * (transitions + e_1 * forc) - lambda_m_t_reshaped

        return tf.reduce_sum(loss)

    # @tf.function add after debug
    def ell_5_6(
        self, lambda_tau_t: tf.Tensor, lambda_tau_tplus: tf.Tensor
    ) -> tf.Tensor:
        """Loss function based on FOC for tau_{t+1}.

        Args:
            lambda_tau_t (tf.Tensor): vector of lagrange multipliers for transformed temperatures at time t
            lambda_tau_tplus (tf.Tensor): vector of lagrange multipliers for transformed temperatures at time t+1

        Returns:
            loss (tf.Tensor)
        """
        # Ensuring that lambda_tau is of correct shape
        lambda_tau_t_reshaped = tf.reshape(lambda_tau_t, (2, 1))
        lambda_tau_tplus_reshaped = tf.reshape(lambda_tau_tplus, (2, 1))

        sigma_transposed = tf.transpose(self.sigma_transition)
        transitions = tf.matmul(sigma_transposed, lambda_tau_tplus_reshaped)

        loss = self.beta * transitions - lambda_tau_t_reshaped
        return tf.reduce_sum(loss)

    # @tf.function add after debug
    def ell_7(
        self,
        k_tplus: tf.Tensor,
        x_t: tf.Tensor,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        tau_1_t: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on lambda{k,t} - budget constraint.

        Args:
            k_tplus (tf.Tensor): log capital in the next time period
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
        consumption = tf.math.log(1 - x_t)
        investment_multiplier = tf.math.log(1 + self.g_k) - tf.math.log(
            self.delta_k + self.g_k
        )

        return production + damages + consumption + investment_multiplier - k_tplus

    # @tf.function add after debug
    def ell_8(self, lambda_E_t: tf.Tensor, E_t: tf.Tensor) -> tf.Tensor:
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
    def ell_9(
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
    def ell_10(
        self,
        value_func_t: tf.Tensor,
        value_func_tplus: tf.Tensor,
        x_t: tf.Tensor,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        tau_1_t: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """Loss function based on Bellman Equation.

        Args:
            value_func_t (tf.Tensor): value function of state at time t
            x_t (tf.Tensor): consumption rate
            E_t (tf.Tensor): emissions at time t
            k_t (tf.Tensor): log capital at time t
            tau_1_t (tf.Tensor): transformed temperature 1
            t (tf.Tensor): timestep

        Returns:
            loss (tf.Tensor)
        """
        x_t_adj = tf.maximum(x_t, 1e-7)  # For avoiding log(0) below

        production = self.equations_of_motion.log_Y_t(k_t, E_t, t)
        damages = -self.xi_0 * tau_1_t + self.xi_0

        return (
            tf.math.log(x_t_adj)
            + production
            + damages
            + self.beta * value_func_tplus
            - value_func_t
        )

    # @tf.function add after debug
    def ell_11(self, x_t: tf.Tensor, lambda_k_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on foc for x_t

        Args:
        x_t (tf.Tensor): Current period's consumption rate.
        lambda_k_t (tf.Tensor): Lagrange multiplier for the budget constraint on capital accumulation.

        Returns:
            loss (tf.Tensor)
        """
        # NOTE: Problem if x_t too close to 1 or 0.
        x_t_adj = tf.raw_ops.ClipByValue(
            x_t, clip_value_min=1e-7, clip_value_max=1 - 1e-7, name=None
        )
        log_x_derivative = 1 / x_t_adj
        marginal_value_of_consumption = -self.beta * (
            (lambda_k_t * self.kappa) / (1 - x_t_adj)
        )
        k_tplus_wrt_x = -lambda_k_t / (1 - x_t_adj)

        return log_x_derivative + marginal_value_of_consumption + k_tplus_wrt_x

    # @tf.function add after debug
    def ell_12(
        self,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        lambda_m_t: tf.Tensor,
        lambda_tau_1_t: tf.Tensor,
        lambda_k_t: tf.Tensor,
        t: tf.Tensor,
    ) -> tf.Tensor:
        """
        Calculates a component of the loss function related to the first-order condition (FOC)
        for emissions at time t.

        Args:
            E_t (tf.Tensor): Emissions at time t.
            k_t (tf.Tensor): Log capital stock at time t.
            lambda_m_t (tf.Tensor): Lagrange multipliers related to the carbon stock.
            lambda_tau_1_t (tf.Tensor): Lagrange multiplier related to the transformed tempreatures.
            lambda_k_t (tf.Tensor): Lagrange multiplier for the budget constraint on capital accumulation.
            t (tf.Tensor): Tensor representing the current time period index.

        Returns:
            tf.Tensor: Loss
        """

        # Get constants from equations of motion
        theta_1_t = self.equations_of_motion.theta_1[t]
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        E_t = tf.minimum(E_t, E_t_BAU)

        # Calculating dlogF/dE_t
        mu_t = 1 - E_t / E_t_BAU
        numerator = theta_1_t * self.theta_2 * tf.pow(mu_t, self.theta_2 - 1)
        denominator = E_t_BAU * (1 - theta_1_t * tf.pow(mu_t, self.theta_2))
        dlogF_dEt = numerator / denominator

        # Calculating dV_tplus/dE_t
        phi_col_1 = self.Phi[:, 0]
        d_V_tplus_dE_t = self.beta * (
            tf.tensordot(phi_col_1, lambda_m_t, axes=1)
            + lambda_tau_1_t * self.sigma_forc * (1 / self.M_pre)
        )

        ell_11 = dlogF_dEt + d_V_tplus_dE_t + lambda_k_t * dlogF_dEt + lambda_m_t[0]
        return ell_11

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
