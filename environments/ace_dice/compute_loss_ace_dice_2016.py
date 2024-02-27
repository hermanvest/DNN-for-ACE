from typing import Tuple, Dict, Any
from environments.ace_dice.equations_of_motion_ace_dice_2016 import (
    Equations_of_motion_Ace_Dice_2016,
)
import tensorflow as tf
import numpy as np


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
    def fischer_burmeister_function(self, a: float, b: float):
        powers = tf.pow(a, 2) + tf.pow(b, 2)
        square_roots = tf.sqrt(powers)

        return a + b - square_roots

    ################ PENALTY BOUNDS ################
    # Penalizing for negative consumption and so on...
    def penalty_bounds_of_policy(self):
        raise NotImplementedError

    ################ INDIVIDUAL LOSS FUNCTIONS ################
    def ell_1(self, lambda_k_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on FOC for k_{t+1}.

        Args:
            lambda_k_t (tf.Tensor): shadow value of capital

        Returns:
            loss (tf.Tensor)
        """
        return self.beta * lambda_k_t * self.kappa - lambda_k_t

    def ell_2_4(self, lambda_m_t: tf.Tensor, lambda_tau_1_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on FOC for M_{t+1}.

        Args:
            lambda_m_t (tf.Tensor): vector of langrange multipliers for the carbon stocks
            lambda_tau_1_t (tf.Tensor): lagrange multiplier for transformed temperature 1 at time t

        Returns:
            loss (tf.Tensor)
        """
        # Ensuring that the lambda vector is of the correct shape
        lambda_m_t_reshaped = tf.reshape(lambda_m_t, (3, 1))
        transitions = tf.matmul(self.Phi, lambda_m_t_reshaped)

        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])

        forc = lambda_tau_1_t * self.sigma_forc * (1 / self.M_pre)

        loss = self.beta * (transitions + e_1 * forc) - lambda_m_t_reshaped

        return tf.reduce_sum(loss)

    def ell_5_6(self, lambda_tau_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on FOC for tau_{t+1}.

        Args:
            lambda_tau_t (tf.Tensor): vector of lagrange multipliers for transformed temperatures at time t

        Returns:
            loss (tf.Tensor)
        """
        # Ensuring that lambda_tau is of correct shape
        lambda_tau_t_reshaped = tf.reshape(lambda_tau_t, (2, 1))

        sigma_transposed = tf.transpose(self.sigma_transition)
        transitions = tf.matmul(sigma_transposed, lambda_tau_t_reshaped)

        loss = self.beta * transitions - lambda_tau_t_reshaped
        return tf.reduce_sum(loss)

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

    def ell_11(self, x_t: tf.Tensor, lambda_k_t: tf.Tensor) -> tf.Tensor:
        """Loss function based on foc for x_t

        Args:
        x_t (tf.Tensor): Current period's consumption rate.
        lambda_k_t (tf.Tensor): Lagrange multiplier for the budget constraint on capital accumulation.

        Returns:
            loss (tf.Tensor)
        """
        log_x_derivative = 1 / x_t
        marginal_value_of_consumption = -self.beta * (
            (lambda_k_t * self.kappa) / (1 - x_t)
        )
        k_tplus_wrt_x = -lambda_k_t / (1 - x_t)

        return log_x_derivative + marginal_value_of_consumption + k_tplus_wrt_x

    def ell_12(self) -> tf.Tensor:
        """Loss function based on foc for x_t

        Returns:
            tf.Tensor: _description_
        """
        pass

    ################ MAIN LOSS FUNCTION CALLED FROM ENV ################
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
        # action variables t
        x_t = a_t[0]
        E_t = a_t[1]
        value_func_t = a_t[2]
        lambda_k_t = a_t[3]
        lambda_m_t_vector = a_t[4:7]  # Indexes look wierd, but are correct
        lambda_tau_t_vector = a_t[7:9]  # Indexes look wierd, but are correct
        lambda_t_BAU = a_t[9]
        lambda_E_t = a_t[10]

        # action variables t+1
        value_func_tplus = a_tplus[2]

        # state variables t
        k_t = s_t[0]
        tau_t_vector = s_t[4:6]  # Indexes look wierd, but are correct
        t = tf.cast(s_t[6], tf.int32)

        # state variables t+1
        k_tplus = s_tplus[0]

        # TODO: This is an abomination. Need to find time to make this prettier.
        loss1 = tf.square(self.ell_1(lambda_k_t))
        loss2_4 = tf.square(self.ell_2_4(lambda_m_t_vector, lambda_tau_t_vector[0]))
        loss5_6 = tf.square(self.ell_5_6(lambda_tau_t_vector))
        loss7 = tf.square(self.ell_7(k_tplus, x_t, E_t, k_t, tau_t_vector[0], t))
        loss8 = tf.square(self.ell_8(lambda_E_t, E_t))
        loss9 = tf.square(self.ell_9(lambda_t_BAU, E_t, k_t, t))
        loss10 = tf.square(
            self.ell_10(
                value_func_t,
                value_func_tplus,
                x_t,
                E_t,
                k_t,
                tau_t_vector[0],
                t,
            )
        )

        total_loss = loss1 + loss2_4 + loss5_6 + loss7 + loss8 + loss9 + loss10
        total_loss_float32 = tf.cast(total_loss, dtype=tf.float32)

        return total_loss_float32
