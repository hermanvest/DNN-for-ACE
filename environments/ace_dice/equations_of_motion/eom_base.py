import tensorflow as tf
from utils.debug import assert_valid
from environments.ace_dice.computation_utils import (
    custom_sigmoid,
    convert_co2_to_c,
)


class Eom_Base:
    """Base class for the equations of motion for ACE with the DICE production function for any version of DICE.

    NOTE: This class assumes that initializations of constants are being made in the sublasses that extend this superclass.
    """

    def __init__(self, t_max: int, mod_version: str) -> None:
        self.t_max = t_max
        self.sigma_transition = self.create_sigma_transitions()
        self.Phi = self.create_Phi_transitions()
        self.E_t_EXO = self.create_e_land(t_max + 1)

        # Note: beta is adjusted for the time step.
        self.beta = tf.exp(-self.timestep * self.prtp)

        self.mod_version = mod_version

    ################ INITIALIZAITON FUNCTIONS ################
    def create_e_land(self, t_max: int) -> tf.Tensor:
        """Computes and returns landuse emissions in GtCO2 per year.

        Args:
            t_max (int): max simulation length

        Returns:
            tf.Tensor: landuse emissions

        Relevant GAMS code:
            eland0         Carbon emissions from land 2015 (GtCO2 per year)  / 2.6  (2016) or 5.9 (2023)    /
            deland         Decline rate of land emissions (per period)       / .115 (2016) or 0.1 (2023)     /
            eland(t) = eland0*(1-deland)**(t.val-1);
        """
        t = tf.range(0, t_max, dtype=tf.float32)

        e_land = self.e_land0 * tf.pow(1 - self.de_land, t)

        return e_land

    def create_sigma_transitions(self) -> tf.Tensor:
        """Used for initializing the transition matrix for temperatures. Assumes that values sigma_up_1, sigma_up_2, and sigma_down_1 are already initialized.

        Returns:
            tf.Tensor: transition matrix for temperatures
        """
        one = tf.constant(1.0, dtype=tf.float32)

        # Calculate the retention rates
        upper_layer_retention = one - self.sigma_up_1 - self.sigma_down_1
        lower_layer_retention = one - self.sigma_up_2

        transition_matrix = tf.stack(
            [
                tf.stack([upper_layer_retention, self.sigma_down_1]),
                tf.stack([self.sigma_up_2, lower_layer_retention]),
            ]
        )

        return transition_matrix

    def create_Phi_transitions(self) -> tf.Tensor:
        """Creates the transition matrix for the carbon reservoirs of a three-layer carbon model, given the Xi vector of coefficients.

        Returns:
            tf.Tensor: The transition matrix for the carbon reservoirs.
        """
        five = tf.constant(5.0, dtype=tf.float32)

        xi_scaled = self.Xi * (self.timestep / five)

        transition_matrix = tf.stack(
            [
                tf.stack([1.0 - xi_scaled[0], xi_scaled[2], 0.0]),
                tf.stack(
                    [xi_scaled[0], 1.0 - xi_scaled[1] - xi_scaled[2], xi_scaled[3]]
                ),
                tf.stack([0.0, xi_scaled[1], 1.0 - xi_scaled[3]]),
            ],
            axis=0,
        )

        return transition_matrix

    ################ HELPER FUNCTIONS ################
    def Y_gross(self, t: int, k_t: tf.Tensor) -> tf.Tensor:
        """Computes Y gross, using log capital.
        Args:
            t (int): time index
            k_t (tf.Tensor): log capital in period t

        Returns:
            Y_t_gross (tf.Tensor): Gross output in trillion USD

        Relevant equation:
            Y_t_gross = A_t ((N_t/1000)^{1-kappa}) K_t^{kappa}

        Relevant GAMS code:
            YGROSS(t)      =E= (AL(t)*(L(t)/1000)**(1-gama))*(K(t)**gama);
        """
        A_t = self.A_t[t]
        N_t = self.N_t[t]
        K_t = tf.math.exp(k_t)  # Need to convert k_t to K_t
        labor_contrib = tf.pow(N_t / 1000, (1 - self.kappa))
        capital_contrib = tf.pow(K_t, self.kappa)

        return A_t * labor_contrib * capital_contrib

    def E_t_BAU(self, t: int, k_t: tf.Tensor) -> tf.Tensor:
        """Returns yearly Business as Usual emissions at time t.

        Relevant equation:
            E_t_BAU = sigma_t * Y_gross_t
        """
        return self.sigma[t] * self.Y_gross(t, k_t)

    def log_Y_t(self, k_t: tf.Tensor, E_t: tf.Tensor, t: int) -> tf.Tensor:
        """Computes log output with abatement costs for time t. Note that this function scales E_t to be in the range (epsilon, E_t_BAU)

        Args:
            k_t (tf.Tensor): log capital for time step t
            E_t (tf.Tensor): unadjusted yearly emisions for time step t

        Returns:
            log Y_t (tf.Tensor): Log of output in trillion USD

        Relevant equation:
            log(Y_t) = log(Y_t_gross) + log(1 - theta_{1,t}(1-E_t/E_t_BAU)^theta_2 )

        Relevant GAMS code:
            abateeq(T)..         ABATECOST(T)   =E= YGROSS(T) * COST1TOT(T) * (MIU(T)**EXPCOST2);
            ygrosseq(t)..        YGROSS(t)      =E= (AL(t)*(L(t)/1000)**(1-gama))*(K(t)**gama);
            yneteq(t)..          YNET(t)        =E= YGROSS(t)*(1-damfrac(t));

            NOTE: Damages are taken care of in the capital equation of motion.
        """
        if self.mod_version == "2016":
            E_t_BAU = self.E_t_BAU(t, k_t)
            E_t_adj = custom_sigmoid(x=E_t, upper_bound=E_t_BAU)
        else:
            E_t_BAU = self.E_t_BAU(t, k_t) + self.E_t_EXO[t]
            E_t_adj = custom_sigmoid(x=E_t, upper_bound=E_t_BAU)

        log_Y_t_gross = tf.math.log(self.Y_gross(t, k_t))

        mu_t = 1 - E_t_adj / E_t_BAU
        abatement_cost = 1 - self.theta_1[t] * tf.pow((mu_t), self.theta_2)
        log_abatement_cost = tf.math.log(abatement_cost)

        result = log_Y_t_gross + log_abatement_cost
        return result

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
        depreciation_factor = tf.math.log(
            (1 + self.g_k * self.timestep) / ((self.delta_k + self.g_k) * self.timestep)
        )

        result = (
            log_Y_t
            + damages
            + log_one_x_t
            + depreciation_factor
            + tf.math.log(self.timestep)
        )
        return result

    def m_tplus(
        self,
        m_t: tf.Tensor,
        E_t: tf.Tensor,
        k_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        """Computes carbon reservoirs for the next period using the equation of motion. Note that this function scales E_t to be in the range (epsilon, E_t_BAU)
        Args:
            m_t (tf.Tensor): The current vector of carbon stocks M_t
            E_t (tf.Tensor): Yearly emissions at current timestep.
            k_t (tf.Tensor): log capital in period t
            t (int): time step
            E_t_EXO (tf.Tensor): Yearly exogenous emissions at current time step

        Returns
            M_{t+1} (tf.Tensor): Carbon stock for all layers in the next time step.
        """
        E_t_BAU = self.E_t_BAU(t, k_t)
        if self.mod_version == "2016":
            E_t_adj = custom_sigmoid(x=E_t, upper_bound=E_t_BAU) + self.E_t_EXO[t]
        else:
            E_t_adj = custom_sigmoid(x=E_t, upper_bound=(E_t_BAU + self.E_t_EXO[t]))

        m_reshaped = tf.reshape(m_t, (3, 1))
        phi_mult_m = tf.matmul(self.Phi, m_reshaped)

        e_1 = tf.constant([1, 0, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[3, 1])

        result = phi_mult_m + (e_1 * (convert_co2_to_c(E_t_adj))) * self.timestep

        return tf.reshape(result, [-1])

    def tau_tplus(
        self,
        tau_t: tf.Tensor,
        m_1_t: tf.Tensor,
        G_t: tf.Tensor = tf.constant(0.0, dtype=tf.float32),
    ) -> tf.Tensor:
        """
        Args:
        - The current vector of transformed temperatures tau
        - Current atmospheric carbon stock M_{1,t}
        - Exogenous non-CO2 ghg G_t (now treated as cumulative sum of exo emisisons for the timesteps in between)

        Returns:
            tau_{t+1} (tf.Tensor): next periods vector of transformed temperatures
        """
        tau_t_reshaped = tf.reshape(tau_t, (2, 1))

        tau_transition = tf.matmul(self.sigma_transition, tau_t_reshaped)
        forcing = self.sigma_forc * ((m_1_t + G_t) / self.M_pre)

        e_1 = tf.constant([1, 0], dtype=tf.float32)
        e_1 = tf.reshape(e_1, shape=[2, 1])

        result = tau_transition + e_1 * forcing
        return tf.reshape(result, [-1])

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
        # For debug purposes
        # assert_valid(a_t, "a_t")
        # assert_valid(s_t, "s_t")

        # state- and action values should be in the same order as in the config
        # 1. get x_t and E_t
        x_t = a_t[0]
        E_t = a_t[1]

        # 2. get all the state values
        k_t = s_t[0]
        m_vector = s_t[1:4]
        tau_vector = s_t[4:6]
        t = (int)(s_t[6].numpy())
        t_plus = s_t[6] + 1  # Needs to be tensor

        # 3. call functions with variables and get state values
        log_Y_t = self.log_Y_t(k_t, E_t, t)
        k_tplus = self.k_tplus(log_Y_t, tau_vector[0], x_t)
        m_plus = self.m_tplus(m_vector, E_t, k_t, t)
        tau_tplus = self.tau_tplus(tau_vector, m_vector[0])

        # 4. return state values for next state
        s_t_plus_tensor = tf.concat(
            [
                tf.reshape(k_tplus, [1]),
                m_plus,
                tau_tplus,
                tf.reshape(t_plus, [1]),
            ],
            axis=0,
        )

        return s_t_plus_tensor
