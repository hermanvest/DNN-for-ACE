from typing import Tuple, Dict, Any
import tensorflow as tf
import numpy as np


class Computeloss:
    def __init__(self, parameters_config: Dict[str, Any]) -> None:
        # Variable initialization based on the config files parameters
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                setattr(self, variable_name, variable_value)

        self.Phi = np.array(self.Phi)
        self.sigma_transition = np.array(self.sigma_transition)

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

    def ell_2_4(self, lambda_m_t_vector: np.ndarray, lambda_tau_1_t: float) -> float:
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

        return self.beta * (transitions + np.matmul(e_1, forc)) - lambda_m_t_vector

    def ell_5_6(self, lambda_tau_t_vector: np.ndarray) -> float:
        """Loss function based on FOC for tau_{t+1}.

        Args:
            var (type): descr

        Returns:
            loss (float)
        """
        sigma_transposed = np.transpose(self.sigma_transition)
        transitions = np.matmul(sigma_transposed, lambda_tau_t_vector)

        return self.beta * (transitions) - lambda_tau_t_vector

    def ell_7(self) -> float:
        """Loss function based on lambda{k,t} - budget constraint.

        Args:
            var (type): descr

        Returns:
            loss (float)
        """

    def ell_8(self) -> float:
        """Loss function based on Fischer-Burmeister function for E_t > 0.

        Args:
            var (type): descr

        Returns:
            loss (float)
        """

    def ell_9(self) -> float:
        """Loss function based on Fischer-Burmeister function for E_t < E_t_BAU.

        Args:
            var (type): descr

        Returns:
            loss (float)
        """

    def ell_10(self) -> float:
        """Loss function based on Bellman Equation.

        Args:
            var (type): descr

        Returns:
            loss (float)
        """

    ################ MAIN LOSS FUNCTION CALLED FROM ENV ################
    def squared_error_for_transition(
        self, s_t: tf.Tensor, a_t: tf.Tensor, s_tplus: tf.Tensor
    ) -> Tuple[float, float]:
        """Returns the squared error for the state transition from s_t to s_tplus, taking action a_t in the environment

        Args:
            s_t (tf.Tensor): state variables at time t
            a_t (tf.Tensor): action variables at time t
            s_tplus (tf.Tensor): state variables at time t+1

        Returns:
            Tuple[float, float]: (squared error without penalty, squared error with penalty)
        """
        # action variables
        x_t = a_t.numpy()[0]
        E_t = a_t.numpy()[1]
        V_t = a_t.numpy()[2]
        lambda_k_t = a_t.numpy()[3]
        lambda_m_vector = a_t.numpy()[4:7]
        lambda_tau_vector = a_t.numpy()[7:9]

        # state variables
        k_t = s_t.numpy()[0]
        m_t_vector = s_t.numpy()[1:4]
        tau_t_vector = s_t.numpy()[4:6]
        t = s_t.numpy()[6]

        # next state variables
        k_tplus = s_tplus.numpy()[0]
        m__tplus_vector = s_tplus.numpy()[1:4]
        tau_tplus_vector = s_tplus.numpy()[4:6]
        t_plus = s_tplus.numpy()[6]

        # start calling individual loss functions
