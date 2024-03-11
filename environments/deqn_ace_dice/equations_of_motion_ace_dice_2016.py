import numpy as np
import tensorflow as tf

from typing import Any, Dict, List
from utils.debug import assert_valid
from environments.deqn_ace_dice.computation_utils import custom_sigmoid


class Equations_of_motion_Ace_Dice_2016:
    """
    Docstring
    """

    def __init__(
        self, t_max: int, states: List, actions: List, parameters_config: Dict[str, Any]
    ) -> None:
        # Initialization based on config file
        # Note that all variables can be found there
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                tensor_value = tf.constant(variable_value, dtype=tf.float32)
                setattr(self, variable_name, tensor_value)

        self.beta = tf.constant(
            (1 / (1 + self.prtp)) ** self.timestep, dtype=tf.float32
        )

        self.states = states
        self.actions = actions
        self.N_t = self.create_N_t(t_max)
        self.A_t = self.create_A_t(t_max)
        self.sigma = self.create_sigma(t_max)
        self.theta_1 = self.create_theta_1(t_max)

    ################ HELPER FUNCITONS ################
    def create_N_t(self, t_max: int) -> tf.Tensor:
        """_summary_

        Args:
            None

        Returns:
            tf.Tensor: A list of the labor inputs we need for all time steps.

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
            labor[t + 1] = labor[t] * (self.popasym / labor[t]) ** self.popadj

        labor_tensor = tf.convert_to_tensor(labor[1:], dtype=tf.float32)

        return labor_tensor

    def create_A_t(self, t_max: int) -> tf.Tensor:
        """
        Creates a list of log total factor productivity we need for all time steps.

        Args:
            None

        Returns:
            tf.Tensor: A list of the log of total factor productivity for all time steps.

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

        labor_tensor = tf.convert_to_tensor(tfp[1:], dtype=tf.float32)

        return labor_tensor

    def create_sigma(self, t_max: int) -> tf.Tensor:
        """
        Computes and returns the CO2-equivalent-emissions output ratio

        Args:
            t (int): time

        Returns:
            tf.Tensor: CO2-equivalent-emissions output ratio sigma_t

        Relevant GAMS code:
            tstep    Years per Period                                     /5       /
            gsigma1  Initial growth of sigma (per year)                   /-0.0152 /
            dsig     Decline rate of decarbonization (per period)         /-0.001  /
            gsig("1")=gsigma1; loop(t,gsig(t+1)=gsig(t)*((1+dsig)**tstep) ;);
            sigma("1")=sig0;   loop(t,sigma(t+1)=(sigma(t)*exp(gsig(t)*tstep)););
        """
        # Initialize growth_sigma and sigma as numpy arrays filled with zeros
        growth_sigma = np.zeros(t_max)
        sigma = np.zeros(t_max)

        # Set initial values
        growth_sigma[0] = self.g_0_sigma
        sigma[0] = self.E_0 / (self.Y_0 * (1 - self.mu_0))

        for t in range(1, t_max):
            growth_sigma[t] = growth_sigma[t - 1] * np.power(
                (1 + self.delta_sigma), self.timestep
            )
            sigma[t] = sigma[t - 1] * np.exp(growth_sigma[t - 1] * self.timestep)

        sigma_tensor = tf.convert_to_tensor(sigma, dtype=tf.float32)
        return sigma_tensor

    def create_theta_1(self, t_max: int) -> tf.Tensor:
        """
        Computes and returns the abatement costs. Note that this method assumes that the vector of sigmas are already created.

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
        theta_1 = np.zeros(t_max)
        for t in range(t_max):
            p_back_t = self.p_back * np.power((1 - self.g_back), t - 1)
            theta_1_t = p_back_t * (self.sigma[t] / (self.theta_2 * 1000))
            theta_1[t] = theta_1_t

        theta_1_tensor = tf.convert_to_tensor(theta_1, dtype=tf.float32)
        return theta_1_tensor

    def Y_gross(self, t: int, k_t: tf.Tensor) -> tf.Tensor:
        """Computes Y gross, using log capital.
        Args:
            t (int): time index
            k_t (tf.Tensor): log capital in period t

        Returns:
            Y_t_gross (tf.Tensor): Gross output in trillion USD

        Relevant equation:
            Y_t_gross = A_t (N_t^{1-kappa} / 1000) K_t^{kappa}
        """
        A_t = self.A_t[t]
        N_t = self.N_t[t]
        K_t = tf.math.exp(k_t)  # Need to convert k_t to K_t
        labor_contrib = tf.pow(N_t, (1 - self.kappa)) / 1000
        capital_contrib = tf.pow(K_t, self.kappa)

        return A_t * labor_contrib * capital_contrib

    def log_Y_gross(self, t: int, k_t: tf.Tensor) -> tf.Tensor:
        """Computes log Y gross using log capital.
        Args:
            t (int): time index
            k_t (tf.Tensor): log capital in period t

        Returns:
            log(Y_t_gross) (tf.Tensor): Log of gross output in trillion USD

        Relevant equation:
            Y_t_gross   = log( A_t * (N_t^(1-kappa) / 1000) * K_t^kappa )
                        = log(A_t) + log( (N_t^(1-kappa)) / 1000) + log(K_t^kappa)
                        = a_t + (1-kappa) log(N_t) - log(1000) + (kappa)k_t
        """
        a_t = tf.math.log(self.A_t[t])
        N_t = self.N_t[t]
        labor_contrib = (1 - self.kappa) * tf.math.log(N_t) - tf.math.log(
            tf.constant(1000.0)
        )
        capital_contrib = self.kappa * k_t

        return a_t + labor_contrib + capital_contrib

    def E_t_BAU(self, t: int, k_t: tf.Tensor) -> tf.Tensor:
        """Returns Business As Usual emissions at time t.

        Relevant equation:
            E_t_BAU = sigma_t * Y_gross_t
        """
        return self.sigma[t] * self.Y_gross(t, k_t)

    def log_Y_t(self, k_t: tf.Tensor, E_t: tf.Tensor, t: int) -> tf.Tensor:
        """Computes log output with abatement costs for time t. Note that this function scales E_t to be in the range (epsilon, E_t_BAU)

        Args:
            k_t (tf.Tensor): log capital for time step t
            E_t (tf.Tensor): emisions for time step t

        Returns:
            log Y_t (tf.Tensor): Log of output in trillion USD

        Relevant equation:
            log(Y_t) = log(Y_t_gross) + log( 1 - theta_{1,t}(|1-E_t/E_t_BAU|)^theta_2 )
        """

        E_t_BAU = self.E_t_BAU(t, k_t)
        E_t_adj = custom_sigmoid(x=E_t, upper_bound=E_t_BAU)

        log_Y_t_gross = self.log_Y_gross(t, k_t)
        mu_t = 1 - E_t_adj / E_t_BAU
        abatement_cost = 1 - self.theta_1[t] * tf.pow((mu_t), self.theta_2)
        log_abatement_cost = tf.math.log(abatement_cost)

        return log_Y_t_gross + log_abatement_cost

    ################ MAIN EQUATIONS ################

    def k_tplus(
        self, log_Y_t: tf.Tensor, tau_1_t: tf.Tensor, x_t: tf.Tensor
    ) -> tf.Tensor:
        """Computes equation of motion for capital. Assumes appropriate bounds on x_t \in (0, 1)
        Args:
            log Y_t (tf.Tensor): Log of production function
            tau_1_t (tf.Tensor): transformed temperatures in layer 1
            x_t (tf.Tensor): consumption rate
            t (int): time index

        Returns:
            k_{t+1} (tf.Tensor): log capital in the next period
        """
        damages = -self.xi_0 * tau_1_t + self.xi_0
        log_one_x_t = tf.math.log(1 - x_t)
        depreciation_factor = tf.math.log(1 + self.g_k) - tf.math.log(
            self.delta_k + self.g_k
        )

        return log_Y_t + damages + log_one_x_t + depreciation_factor

    def m_plus(
        self, m_t: tf.Tensor, E_t: tf.Tensor, k_t: tf.Tensor, t: int
    ) -> tf.Tensor:
        """Computes carbon reservoirs for the next period using the equation of motion. Note that this function scales E_t to be in the range (epsilon, E_t_BAU)
        Args:
            m_t (tf.Tensor): The current vector of carbon stocks M_t
            E_t (tf.Tensor): All emissions at current timestep, including exogenous emissions
            k_t (tf.Tensor): log capital in period t
            t (int): time step

        Returns
            M_{t+1} (tf.Tensor): Carbon stock for all layers in the next time step.
        """
        E_t_BAU = self.E_t_BAU(t, k_t)
        E_t_adj = custom_sigmoid(x=E_t, upper_bound=E_t_BAU)

        m_reshaped = tf.reshape(m_t, (3, 1))
        phi_mult_m = tf.matmul(self.Phi, m_reshaped)

        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])

        return phi_mult_m + e_1 * E_t_adj

    def tau_1plus(
        self, tau_t: tf.Tensor, m_1_t: tf.Tensor, G_t: tf.Tensor = 0.0
    ) -> tf.Tensor:
        """
        Args:
        - The current vector of transformed temperatures tau
        - Current atmospheric carbon stock M_{1,t}
        - Exogenous non-CO2 ghg G_t

        Returns: tau_{1,t+1}
        """
        sigma_transition_row_1 = self.sigma_transition[0, :]

        temp_transitions = tf.tensordot(sigma_transition_row_1, tau_t, axes=1)
        forcing = self.sigma_forc * ((m_1_t + G_t) / self.M_pre)

        return temp_transitions + forcing

    def tau_2plus(self, tau_t: tf.Tensor) -> tf.Tensor:
        """
        Args:
        - Transition matrix for temperatures
        - The current vector of transformed temperatures tau
        - Sigma^forc

        Returns: tau_{2,t+1}
        """
        sigma_transition_row_2 = self.sigma_transition[1, :]
        return tf.tensordot(sigma_transition_row_2, tau_t, axes=1)

    def update_state(self, s_t: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
        """Updates the state based on current state and action tensors.

        Args:
            s_t (tf.Tensor): The current state tensor.
            a_t (tf.Tensor): The action tensor.

        Returns:
            tf.Tensor: The updated state tensor.

        Raises:
            TypeError: If either s_t or a_t is not of dtype tf.float32.
        """
        assert_valid(a_t, "a_t")
        assert_valid(s_t, "s_t")

        # state- and action values should be in the same order as in the config
        # 1. get x_t and E_t
        x_t = a_t[0]
        E_t = a_t[1]

        # 2. get all the state values
        k_t = s_t[0]
        m_vector = s_t[1:4]
        tau_vector = s_t[4:6]
        t = (int)(s_t[6].numpy())
        t_plus = s_t[6] + 1

        # 3. call functions with variables and get state values
        # how do I get G_t?
        G_t = 0

        log_Y_t = self.log_Y_t(k_t, E_t, t)
        k_tplus = self.k_tplus(log_Y_t, tau_vector[0], x_t)
        m_plus = tf.reshape(self.m_plus(m_vector, E_t, k_t, t), [-1])  # Now shape = [3]
        # m_1plus = self.m_1plus(m_vector, E_t, k_t, t)
        # m_2plus = self.m_2plus(m_vector)
        # m_3plus = self.m_3plus(m_vector)
        tau_1plus = self.tau_1plus(tau_vector, m_vector[0], G_t)
        tau_2plus = self.tau_2plus(tau_vector)

        # 4. return state values for next state
        # s_t_plus = [k_tplus, m_1plus, m_2plus, m_3plus, tau_1plus, tau_2plus, t_plus]
        # s_t_plus_tensor = tf.convert_to_tensor(s_t_plus, dtype=tf.float32)
        s_t_plus_tensor = tf.concat(
            [
                tf.reshape(k_tplus, [1]),
                m_plus,
                tf.reshape(tau_1plus, [1]),
                tf.reshape(tau_2plus, [1]),
                tf.reshape(t_plus, [1]),
            ],
            axis=0,
        )

        return s_t_plus_tensor
