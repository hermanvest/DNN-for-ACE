from typing import Any, Dict


class DICE_Prod_Stateupdater:
    """
    The StateUpdater class is responsible for updating various state variables.
    It utilizes a set of constants and current state parameters to calculate the next state.

    The class methods include calculations for next-state values of technology, capital levels, carbon stocks,
    and transformed temperatures, among others. Each method requires specific input parameters relevant to the
    component it updates and returns the calculated next state value.

    Attributes:
        constants (Dict[str, Any]): A dictionary of constants used in state update calculations. These constants are defined in the config files
    """

    def __init__(self, constants: Dict[str, Any]) -> None:
        pass

    def log_f_t(a_t: float, k_t: float, N_t: float, E_t: float) -> float:
        """
        Needs the following in order to return next state:
        - Technology a_t
        - Current capital level k_t
        - Population N_t
        - Emissions E_t

        Returns: log F_t
        """

    def k_tplus() -> float:
        """
        Needs the following in order to return next state:
        - Y_t = F_t(A_t,N_t,K_t,E_t)
        - xi_0
        - tau_{1,t}
        - x_t
        - g_{k,t}
        - delta_k

        Returns: k_{t+1}
        """

    def m_1plus() -> float:
        """
        Needs the following in order to return next state:
        - First row of Phi
        - The current vector of carbon stocks M_t
        - All emissions at current timestep, including exogenous emissions

        Returns: M_{1,t+1}
        """

    def m_2plus() -> float:
        """
        Needs the following in order to return next state:
        - Second row of Phi
        - The current vector of carbon stocks M_t

        Returns: M_{2,t+1}
        """

    def m_3plus() -> float:
        """
        Needs the following in order to return next state:
        - Third row of Phi
        - The current vector of carbon stocks M_t

        Returns: M_{3,t+1}
        """

    def tau_1plus() -> float:
        """
        Needs the following in order to return next state:
        - Transition matrix for temperatures
        - The current vector of transformed temperatures tau
        - Sigma^forc
        - Current atmospheric carbon stock M_{1,t}
        - Preindustrial carbon stock M_{pre}
        - Exogenous non-CO2 ghg G_t

        Returns: tau_{1,t+1}
        """

    def tau_2plus() -> float:
        """
        Needs the following in order to return next state:
        - Transition matrix for temperatures
        - The current vector of transformed temperatures tau
        - Sigma^forc

        Returns: tau_{2,t+1}
        """
