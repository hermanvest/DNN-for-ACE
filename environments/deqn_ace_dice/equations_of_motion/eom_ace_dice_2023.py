import numpy as np
import tensorflow as tf

from typing import Any, Dict, List
from environments.deqn_ace_dice.equations_of_motion.eom_base import Eom_Base


class Eom_Ace_Dice_2023(Eom_Base):
    """Sublass extending the base class for the equations of motion for ACE with the DICE produciton function. This class initializes all constants from a configuration file and does the specific initializations for labor, tfp, carbon intensity of produciton and abatement costs."""

    def __init__(
        self, t_max: int, states: List, actions: List, parameters_config: Dict[str, Any]
    ) -> None:
        # Initialization based on config file
        # Note that all variables can be found there
        for _, section_value in parameters_config.items():
            for variable_name, variable_value in section_value.items():
                tensor_value = tf.constant(variable_value, dtype=tf.float32)
                setattr(self, variable_name, tensor_value)

        self.states = states
        self.actions = actions

        self.N_t = self.create_N_t(t_max)
        self.A_t = self.create_A_t(t_max)
        self.sigma, self.theta_1 = self.create_sigma_theta_1(t_max)
        self.E_t_EXO = self.create_e_land(t_max)

        super().__init__(t_max)

    ################ INITIALIZAITON FUNCTIONS ################
    def create_N_t(self, t_max: int) -> tf.Tensor:
        """_summary_

        Args:
            t_max (int): Max number of episodes that are to be simulated.

        Returns:
            tf.Tensor: A list of the labor inputs we need for all time steps.

        Relevant GAMS code:
            L("1") = pop1; loop(t, L(t+1)=L(t););
            loop(t, L(t+1)  = L(t)*(popasym/L(t))**popadj ;);
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
            t_max (int): Max number of episodes that are to be simulated.

        Returns:
            tf.Tensor: A list of the log of total factor productivity for all time steps.

        Relevant GAMS code:
            gA(t) = gA1*exp(-delA*5*((t.val-1)));
            aL("1") = AL1;
            loop(t, aL(t+1)=aL(t)/((1-gA(t))););
        """
        tfp = np.zeros(t_max + 1)
        tfp[1] = self.a0

        # Pre-calculate the ga values
        t_values = np.arange(1, t_max + 1)
        ga = self.ga0 * np.exp(-self.dela * self.timestep * (t_values - 1))

        # Calculate TFP levels
        for t in range(1, t_max):
            tfp[t + 1] = tfp[t] / (1 - ga[t])

        labor_tensor = tf.convert_to_tensor(tfp[1:], dtype=tf.float32)

        return labor_tensor

    def create_sigma_theta_1(self, t_max: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Computes and returns a tuple with the CO2-equivalent-emissions output ratio and abatement costs.

        Args:
            t_max (int): Max number of episodes that are to be simulated.

        Returns:
            tf.Tensor: CO2-equivalent-emissions output ratio sigma_t
            tf.Tensor: abatement costs theta_1

        Relevant GAMS code:
            pbacktime(t)    = pback2050*exp(-5*(.01)*(t.val-7));
            pbacktime(t)$(t.val > 7) = pback2050*exp(-5*(.001)*(t.val-7));

            gsig(t) = min(gsigma1*delgsig **((t.val-1)),asymgsig);
            sig1            = e1/(q1*(1-miu1)); sigma("1") = sig1;
            loop(t, sigma(t+1)  = sigma(t)*exp(5*gsig(t)););

            emissrat(t) = emissrat2020
                +((emissrat2100-emissrat2020)/16)*(t.val-1)$(t.val le 16)
                +((emissrat2100-emissrat2020))$(t.val ge 17);
            sigmatot(t) = sigma(t)*emissrat(t);
            cost1tot(t) = pbacktime(T)*sigmatot(T)/expcost2/1000;
        """
        t_values = tf.range(1, t_max + 1, dtype=tf.float32)

        # pbacktime(t) calculation
        pbacktime = tf.where(
            t_values > 7,
            self.pback2050 * tf.exp(-self.timestep * 0.001 * (t_values - 7)),
            self.pback2050 * tf.exp(-self.timestep * 0.01 * (t_values - 7)),
        )

        # gsig(t) calculation
        gsig = tf.minimum(self.gsigma1 * self.delgsig ** (t_values - 1), self.asymgsig)

        # sigma(t+1) calculation
        sigma_initial = self.E_1 / (self.Y_1 * (1 - self.mu_1))

        sigma_change_factors = tf.exp(5 * gsig)
        sigma_with_initial = tf.concat(
            [[sigma_initial], sigma_change_factors[:-1]], axis=0
        )
        sigma = tf.math.cumprod(sigma_with_initial, axis=0)

        # emissrat(t) calculation
        linear_part = self.emissrat2020 + (
            (self.emissrat2100 - self.emissrat2020) / 16
        ) * (t_values - 1)
        emissrat = tf.where(t_values <= 16, linear_part, self.emissrat2100)

        # sigmatot(t) calculation
        sigmatot = sigma * emissrat

        # cost1tot(t) calculation
        theta_1 = pbacktime * sigmatot / (self.theta_2 * 1000)

        return sigmatot, theta_1

    def create_e_land(self, t_max: int) -> tf.Tensor:
        """Computes and returns landuse emissions in GtCO2 per year.

        Args:
            t_max (int): max simulation length

        Returns:
            tf.Tensor: landuse emissions

        Relevant GAMS code:
            eland0         Carbon emissions from land 2015 (GtCO2 per year)  / 5.9    /
            deland         Decline rate of land emissions (per period)       / .1     /
            eland(t) = eland0*(1-deland)**(t.val-1);
        """
        t = tf.range(1, t_max + 1, dtype=tf.float32)

        e_land = self.e_land0 * (1 - self.de_land) ** (t)

        return e_land
