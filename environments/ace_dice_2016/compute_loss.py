from typing import Tuple, Dict, Any
from environments.ace_dice_2016.equations_of_motion import Equations_of_motion_Ace_Dice
import tensorflow as tf
import numpy as np


class Computeloss:
    def __init__(
        self,
        parameters_config: Dict[str, Any],
        equations_of_motion: Equations_of_motion_Ace_Dice,
    ) -> None:
        # Variable initialization based on the config files parameters
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                setattr(self, variable_name, variable_value)

        self.Phi = np.array(self.Phi)
        self.sigma_transition = np.array(self.sigma_transition)
        self.equations_of_motion = equations_of_motion

    ################ HELPER FUNCITONS ################
    def fischer_burmeister_function(self, a: float, b: float):
        powers = np.power(a, 2) + np.power(b, 2)
        square_roots = np.sqrt(powers)

        return a + b - square_roots

    ################ PENALTY BOUNDS ################
    # Penalizing for negative consumption and so on...
    def penalty_bounds_of_policy(self):
        raise NotImplementedError

    ################ INDIVIDUAL LOSS FUNCTIONS ################
    def ell_1(self, lambda_k_t: float) -> float:
        """Loss function based on FOC for k_{t+1}.

        Args:
            lambda_k_t (float): shadow value of capital

        Returns:
            loss (float)
        """
        return self.beta * lambda_k_t * self.kappa - lambda_k_t

    def ell_2_4(
        self, lambda_m_t_vector: np.ndarray, lambda_tau_1_t: float
    ) -> np.ndarray:
        """Loss function based on FOC for M_{t+1}.

        Args:
            lambda_m_t_vector (np.ndarray): vector of langrange multipliers for the carbon stocks
            lambda_tau_1_t (float): lagrange multiplier for transformed temperature 1 at time t

        Returns:
            loss (float)
        """
        transitions = np.matmul(self.Phi, lambda_m_t_vector)
        e_1 = np.array([1, 0, 0])
        forc = lambda_tau_1_t * self.sigma_forc * (1 / self.M_pre)

        return self.beta * (transitions + e_1 * forc) - lambda_m_t_vector

    def ell_5_6(self, lambda_tau_t_vector: np.ndarray) -> np.ndarray:
        """Loss function based on FOC for tau_{t+1}.

        Args:
            lambda_tau_t_vector (np.ndarray): vector of lagrange multipliers for transformed temperatures at time t

        Returns:
            loss (float)
        """
        sigma_transposed = np.transpose(self.sigma_transition)
        transitions = np.matmul(sigma_transposed, lambda_tau_t_vector)

        return self.beta * (transitions) - lambda_tau_t_vector

    def ell_7(
        self, k_tplus: float, x_t: float, E_t: float, k_t: float, tau_1_t: float, t: int
    ) -> float:
        """Loss function based on lambda{k,t} - budget constraint.

        Args:
            k_tplus (float): log capital in the next time period
            x_t (float): consumption rate
            E_t (float): emissions at time t
            k_t (float): log capital at time t
            tau_1_t (float): transformed temperature 1
            t (int): timestep

        Returns:
            loss (float)
        """
        production = self.equations_of_motion.log_Y_t(k_t, E_t, t)
        damages = -self.xi_0 * tau_1_t + self.xi_0
        consumption = np.log(1 - x_t)
        investment_multiplier = np.log(1 + self.g_k) - np.log(self.delta_k + self.g_k)

        return production + damages + consumption + investment_multiplier - k_tplus

    def ell_8(self, lambda_E_t: float, E_t: float) -> float:
        """Loss function based on Fischer-Burmeister function for E_t \geq 0.

        Args:
            lambda_E_t (float): lagrange multiplier for the constraint on E_t \geq 0
            E_t (float): emissions at time t

        Returns:
            loss (float)
        """
        fb = self.fischer_burmeister_function(lambda_E_t, E_t)
        return fb

    def ell_9(self, lambda_t_BAU: float, E_t: float, k_t: float, t: int) -> float:
        """Loss function based on Fischer-Burmeister function for E_t \leq E_t_BAU.

        Args:
            lambda_t_BAU (flaot): lagrange multiplier for the condition that E_t \leq E_t_BAU
            E_t (float): emissions at time t
            k_t (float): log capital at time t
            t (int): timestep

        Returns:
            loss (float)
        """
        E_t_BAU = self.equations_of_motion.E_t_BAU(t, k_t)
        emissions_diff = E_t_BAU - E_t
        return self.fischer_burmeister_function(lambda_t_BAU, emissions_diff)

    def ell_10(
        self,
        value_func_t: float,
        value_func_tplus: float,
        x_t: float,
        E_t: float,
        k_t: float,
        tau_1_t: float,
        t: int,
    ) -> float:
        """Loss function based on Bellman Equation.

        Args:
            value_func_t (float): value function of state at time t
            x_t (float): consumption rate
            E_t (float): emissions at time t
            k_t (float): log capital at time t
            tau_1_t (float): transformed temperature 1
            t (int): timestep

        Returns:
            loss (float)
        """
        production = self.equations_of_motion.log_Y_t(k_t, E_t, t)
        damages = -self.xi_0 * tau_1_t + self.xi_0
        # Problem: how do i get the value function of the next state?
        # Solution?: do another prediction from s_tplus to obtain v_tplus

        return (
            np.log(x_t) + production + damages + value_func_tplus - value_func_t
        )  # need to add beta*v_{t+1}

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
        x_t = a_t.numpy()[0]
        E_t = a_t.numpy()[1]
        value_func_t = a_t.numpy()[2]
        lambda_k_t = a_t.numpy()[3]
        lambda_m_t_vector = a_t.numpy()[4:7]  # Indexes look wierd, but are correct
        lambda_tau_t_vector = a_t.numpy()[7:9]  # Indexes look wierd, but are correct
        lambda_t_BAU = a_t.numpy()[9]
        lambda_E_t = a_t.numpy()[10]

        # action variables t+1
        value_func_tplus = a_tplus.numpy()[2]

        # state variables t
        k_t = s_t.numpy()[0]
        tau_t_vector = s_t.numpy()[4:6]  # Indexes look wierd, but are correct
        t = (int)(s_t.numpy()[6])

        # state variables t+1
        k_tplus = s_tplus.numpy()[0]

        losses = np.array([], dtype=float)
        losses = np.concatenate(
            (
                losses,
                np.array([self.ell_1(lambda_k_t)]),
                self.ell_2_4(lambda_m_t_vector, lambda_tau_t_vector[0]),
                self.ell_5_6(lambda_tau_t_vector),
                np.array([self.ell_7(k_tplus, x_t, E_t, k_t, tau_t_vector[0], t)]),
                np.array([self.ell_8(lambda_E_t, E_t)]),
                np.array([self.ell_9(lambda_t_BAU, E_t, k_t, t)]),
                np.array(
                    [
                        self.ell_10(
                            value_func_t,
                            value_func_tplus,
                            x_t,
                            E_t,
                            k_t,
                            tau_t_vector[0],
                            t,
                        )
                    ]
                ),
            )
        )

        squared_losses_sum = np.sum(losses**2)
        return squared_losses_sum
