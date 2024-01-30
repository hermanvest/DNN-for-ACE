from typing import Any, Dict
import numpy as np


class DICE_Prod_Stateupdater:
    """
    The DICE_Prod_Stateupdater class is responsible for updating various state variables.
    It utilizes a set of constants and current state parameters to calculate the next state.

    The class methods include calculations for next-state values of technology, capital levels, carbon stocks,
    and transformed temperatures, among others. Each method requires specific input parameters relevant to the
    component it updates and returns the calculated next state value.

    Attributes:
        constants (Dict[str, Any]): A dictionary of constants used in state update calculations. These constants are defined in the config files
    """

    def __init__(self, constants: Dict[str, Any]) -> None:
        # Initialization based on config file
        # Note that all variables can be found there
        for section_key, section_value in constants.items():
            for variable_name, variable_value in section_value.items():
                setattr(self, variable_name, variable_value)

    # --- HELPER METHODS ---
    def _theta_1_t(self, t: int) -> float:
        """
        Computes and returns the abatement costs.

        Returns:
            float: Abatement costs Theta_{1,t}
        """
        backstop = self.p_0_back * (1 + np.exp(-self.g_back * t))
        carbon = 1000 * self.c2co2 * self._sigma_t(t)

        return (backstop * carbon) / self.theta_2

    def _sigma_t(self, t: int) -> float:
        """
        Computes and returns the carbon intensity sigma_t

        Args:
            t (int): time

        Returns:
            float: carbon intensity sigma_t
        """
        step_intensity = (self.delta_t * self.g_0_sigma) / (
            np.log(1 + self.delta_t * self.delta_sigma)
        )
        decline = np.power(1 + self.delta_t * self.delta_sigma, t) - 1

        return self.sigma_0 * np.exp(step_intensity * decline)

    # --- MAIN EQUATIONS ---

    def log_f_t(
        self, a_t: float, k_t: float, N_t: float, E_t: float, E_t_BAU: float, t: int
    ) -> float:
        """
        Args:
        - Technology: a_t
        - Current capital level: k_t
        - Population: N_t
        - Emissions: E_t and E_t_BAU (Business As Usual
        - Abatement cost reduction over time: Theta_{1,t}

        Returns: log F_t
        """
        capital_contrib = self.kappa * k_t
        labor_contrib = (1 - self.kappa) * np.log(N_t)
        energy_sector = np.log(
            1 - self._theta_1_t(t) * np.power((1 - E_t / E_t_BAU), self.theta_2)
        )

        return a_t + capital_contrib + labor_contrib + energy_sector

    def k_tplus(
        self,
        Y_t: float,
        tau_1_t: float,
        x_t: float,
    ) -> float:
        """
        Args:
        - Y_t = F_t(A_t,N_t,K_t,E_t)
        - tau_{1,t}
        - x_t

        Returns: k_{t+1}
        """
        damages = -self.xi_0 * tau_1_t + self.xi_0
        log_one_x_t = np.log(1 - x_t)
        depreciation_factor = np.log(1 + self.g_k) - np.log(self.delta_k + self.g_k)

        return Y_t + damages + log_one_x_t + depreciation_factor

    def m_1plus(self, m_t: np.array, E_t: float) -> float:
        """
        Args:
        - The current vector of carbon stocks M_t
        - All emissions at current timestep, including exogenous emissions

        Returns: M_{1,t+1}
        """
        phi_row_1 = np.array(self.Phi[0][:])
        phi_dot_m = np.dot(phi_row_1, m_t)

        return phi_dot_m + E_t

    def m_2plus(self, m_t: np.array) -> float:
        """
        Args:
        - The current vector of carbon stocks M_t

        Returns: M_{2,t+1}
        """
        phi_row_2 = np.array(self.Phi[1][:])
        return np.dot(phi_row_2, m_t)

    def m_3plus(self, m_t: np.array) -> float:
        """
        Args:
        - The current vector of carbon stocks M_t

        Returns: M_{3,t+1}
        """

        phi_row_3 = np.array(self.Phi[2][:])
        return np.dot(phi_row_3, m_t)

    def tau_1plus(self, tau_t: np.array, m_1_t: float, G_t: float = 0.0) -> float:
        """
        Args:
        - The current vector of transformed temperatures tau
        - Current atmospheric carbon stock M_{1,t}
        - Exogenous non-CO2 ghg G_t

        Returns: tau_{1,t+1}
        """
        sigma_row_1 = np.array(self.sigma[0][:])

        temp_transitions = np.dot(sigma_row_1, tau_t)
        forcing = self.sigma_forc * ((m_1_t + G_t) / self.M_pre)

        return temp_transitions + forcing

    def tau_2plus(self, tau_t: np.array) -> float:
        """
        Args:
        - Transition matrix for temperatures
        - The current vector of transformed temperatures tau
        - Sigma^forc

        Returns: tau_{2,t+1}
        """
        sigma_row_2 = np.array(self.sigma[1][:])
        return np.dot(sigma_row_2, tau_t)
