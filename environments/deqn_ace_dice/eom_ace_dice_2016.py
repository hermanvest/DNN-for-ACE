import numpy as np
import tensorflow as tf

from typing import Any, Dict, List
from utils.debug import assert_valid
from environments.deqn_ace_dice.computation_utils import custom_sigmoid
from environments.deqn_ace_dice.eom_base import Eom_Base


class Eom_Ace_Dice_2016(Eom_Base):
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
        self.sigma_transition = self.create_sigma_transitions()
        self.N_t = self.create_N_t(t_max)
        self.A_t = self.create_A_t(t_max)
        self.sigma = self.create_sigma(t_max)
        self.theta_1 = self.create_theta_1(t_max)

        super().__init__()

    ################ INITIALIZAITON FUNCTIONS ################
    def create_sigma_transitions(self) -> tf.Tensor:
        """Used for initializing the transition matrix for temperatures. Assumes that values sigma_up_1, sigma_up_2, and sigma_down_1 are already initialized.

        Returns:
            tf.Tensor: transition matrix for temperatures
        """
        # Ensure that constants are also tensors for consistent operations
        one_tensor = tf.constant(1.0, dtype=tf.float32)

        # Calculate the retention rates explicitly using TensorFlow operations
        upper_layer_retention = tf.subtract(
            tf.subtract(one_tensor, self.sigma_up_1), self.sigma_down_1
        )
        lower_layer_retention = tf.subtract(one_tensor, self.sigma_up_2)

        # Construct the transition matrix explicitly
        # Using tf.stack to ensure proper tensor structure
        transition_matrix = tf.stack(
            [
                tf.stack([upper_layer_retention, self.sigma_down_1]),
                tf.stack([self.sigma_up_2, lower_layer_retention]),
            ]
        )

        return transition_matrix

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