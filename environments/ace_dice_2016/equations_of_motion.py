from typing import Any, Dict, List
import numpy as np
import tensorflow as tf


class Equations_of_motion_Ace_Dice:
    """
    The Equations_of_motion_Ace_Dice class is responsible for updating various state variables.
    It utilizes a set of constants and current state parameters to calculate the next state.

    The class methods include calculations for next-state values of technology, capital levels, carbon stocks,
    and transformed temperatures, among others. Each method requires specific input parameters relevant to the
    component it updates and returns the calculated next state value.

    Attributes:
        constants (Dict[str, Any]): A dictionary of constants used in state update calculations. These constants are defined in the config files
    """

    def __init__(
        self, t_max: int, states: List, actions: List, parameters_config: Dict[str, Any]
    ) -> None:
        # Initialization based on config file
        # Note that all variables can be found there
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                setattr(self, variable_name, variable_value)

        self.states = states
        self.actions = actions
        self.N_t = self.create_N_t(t_max)
        self.a_t = self.create_a_t(t_max)
        self.sigma = self.create_sigma(t_max)
        self.theta_1 = self.create_theta_1(t_max)

    def update_state(self, s_t: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
        # state- and action values should be in the same order as in the config
        # 1. get x_t and E_t
        x_t = a_t.numpy()[0]
        E_t = a_t.numpy()[1]

        # 2. get all the state values
        k_t = s_t.numpy()[0]
        m_vector = s_t.numpy()[1:4]
        tau_vector = s_t.numpy()[4:6]
        t = s_t.numpy()[6]

        # 3. call functions with variables and get state values
        # how do I get G_t?
        G_t = 0

        log_Y_t = self.log_Y_t(k_t, E_t, t)
        k_tplus = self.k_tplus(log_Y_t, tau_vector[0], x_t)
        m_1plus = self.m_1plus(m_vector, E_t)
        m_2plus = self.m_2plus(m_vector)
        m_3plus = self.m_3plus(m_vector)
        tau_1plus = self.tau_1plus(tau_vector, m_vector[0], G_t)
        tau_2plus = self.tau_2plus(tau_vector)

        # 4. return state values for next state
        s_t_plus = [k_tplus, m_1plus, m_2plus, m_3plus, tau_1plus, tau_2plus, t + 1]

        s_t_plus_tensor = tf.stack(s_t_plus)
        return s_t_plus_tensor

    # --- HELPER METHODS ---
    def create_N_t(self, t_max) -> np.ndarray:
        """_summary_

        Args:
            None

        Returns:
            np.ndarray: A list of the labor inputs we need for all time steps.

        Relevant GAMS code:
            set        t  Time periods (5 years per period)           /1*100   /
            pop0     Initial world population 2015 (millions)         /7403    /
            popadj   Growth rate to calibrate to 2050 pop projection  /0.134   /
            popasym  Asymptotic population (millions)                 /11500   /
            l("1") = pop0;
            loop(t, l(t+1)=l(t););
            loop(t, l(t+1)=l(t)*(popasym/L(t))**popadj ;);
        """
        labor = np.zeros(t_max + 1)
        labor[1] = self.pop0

        for t in range(1, t_max):
            labor[t + 1] = labor[t] * (self.popasym / labor[t]) ** self.popaj

        return labor[1:]

    def create_a_t(self, t_max) -> np.ndarray:
        """
        Creates a list of log total factor productivity we need for all time steps.

        Args:
            None

        Returns:
            np.ndarray: A list of the log of total factor productivity for all time steps.

        Relevant GAMS code:
            set        t  Time periods (5 years per period)           /1*100   /
            dela     Decline rate of TFP per 5 years                  /0.005   /
            ga0      Initial growth rate for TFP per 5 years          /0.076   /
            a0       Initial level of total factor productivity       /5.115   /
            ga(t)=ga0*exp(-dela*5*((t.val-1)));
            al("1") = a0; loop(t, al(t+1)=al(t)/((1-ga(t))););
        """
        tfp = np.zeros(t_max + 1)
        tfp[1] = self.a0

        # Pre-calculate the ga values for efficiency
        t_values = np.arange(1, t_max + 1)
        ga = self.ga0 * np.exp(-self.dela * self.timestep * (t_values - 1))

        # Calculate TFP levels
        for t in range(1, t_max):
            tfp[t + 1] = tfp[t] / (1 - ga[t])

        return np.log(tfp[1:])

    def y_gross(self, t: int, k_t: float) -> float:
        """_summary_
        Args:

        Returns: Y_t gross in trillion USD

        Relevant equation:
            Y_t = A_t (N_t^{1-kappa} / 1000) K_t^{kappa}
        """

        A_t = np.exp(self.a_t[t])
        N_t = self.N_t[t]

        return (
            A_t * (np.power(N_t, (1 - self.kappa)) / 1000) * np.power(k_t, self.kappa)
        )

    def create_sigma(self, t_max: int) -> tf.Tensor:
        """
        Computes and returns the carbon intensity sigma_t

        Args:
            t (int): time

        Returns:
            float: carbon intensity sigma_t

        Relevant GAMS code:
            tstep    Years per Period                                     /5       /
            gsigma1  Initial growth of sigma (per year)                   /-0.0152 /
            dsig     Decline rate of decarbonization (per period)         /-0.001  /
            gsig("1")=gsigma1; loop(t,gsig(t+1)=gsig(t)*((1+dsig)**tstep) ;);
            sigma("1")=sig0;   loop(t,sigma(t+1)=(sigma(t)*exp(gsig(t)*tstep)););
        """
        sigma = []
        for t in range(t_max):
            step_intensity = (self.delta_t * self.g_0_sigma) / (
                np.log(1 + self.delta_t * self.delta_sigma)
            )
            decline = np.power(1 + self.delta_t * self.delta_sigma, t) - 1
            sigma.append(self.sigma_0 * np.exp(step_intensity * decline))

        return tf.stack(sigma)

    def theta_1_t(self, t_max: int) -> tf.Tensor:
        """
        Computes and returns the abatement costs.

        Args:
            t (int): time

        Returns:
            Tensor: Abatement costs for all timesteps Theta_{1,t}

        Relevant GAMS code:
            expcost2  Exponent of control cost function               / 2.6  /
            pback     Cost of backstop 2010$ per tCO2 2015            / 550  /
            gback     Initial cost decline backstop cost per period   / .025 /
            pbacktime(t)=pback*(1-gback)**(t.val-1);
            cost1(t) = pbacktime(t)*sigma(t)/expcost2/1000;
        """
        theta_1 = []
        for t in range(t_max):
            p_back_t = self.p_back * np.power((1 - self.g_back), t - 1)
            theta_1_t = p_back_t * (self.sigma[t] / (self.theta2 * 1000))
            theta_1.append(theta_1_t)
        return tf.stack(theta_1)

    def E_t_BAU(self, t: int, k_t: float) -> float:
        """Returns Business As Usual emissions at time t.

        Relevant equation:
            E_t_BAU = sigma_t * Y_gross_t
        """
        return self.sigma_t(t) * self.y_gross(t, k_t)

    def log_Y_t(self, k_t: float, E_t: float, t: int) -> float:
        """Computes log output with abatement costs for time t.

        Args:
            k_t (float): log capital for time step t
            E_t (float): emisions for time step t

        Returns:
            log Y_t (float): logarithm of output Y_t

        Relevant equation:
            log(Y_t) = log(Y_t_gross) + log( 1 - theta_{1,t}(1-E_t/E_t_BAU)^theta_2 )
        """
        log_Y_t_gross = np.log(self.y_gross(t, k_t))
        abatement_cost = 1 - self.theta_1[t] * np.power(
            (1 - E_t / self.E_t_BAU(t, k_t))
        )
        log_abatement_cost = np.log(abatement_cost)
        return log_Y_t_gross + log_abatement_cost

    # --- MAIN EQUATIONS ---

    def k_tplus(
        self,
        log_Y_t: float,
        tau_1_t: float,
        x_t: float,
    ) -> float:
        """
        Args:
        - log Y_t = log F_t(A_t,N_t,K_t,E_t)
        - tau_{1,t}
        - x_t

        Returns: k_{t+1}
        """
        damages = -self.xi_0 * tau_1_t + self.xi_0
        log_one_x_t = np.log(1 - x_t)
        depreciation_factor = np.log(1 + self.g_k) - np.log(self.delta_k + self.g_k)

        return log_Y_t + damages + log_one_x_t + depreciation_factor

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
