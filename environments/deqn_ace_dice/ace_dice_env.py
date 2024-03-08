import tensorflow as tf
import numpy as np

from typing import Any, Dict
from environments.abstract_environment import Abstract_Environment
from environments.deqn_ace_dice.equations_of_motion_ace_dice_2016 import (
    Equations_of_motion_Ace_Dice_2016,
)
from environments.deqn_ace_dice.compute_loss_ace_dice_2016 import (
    Computeloss_Ace_Dice_2016,
)


class Ace_dice_env(Abstract_Environment):
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        # Extracting the specific configurations
        general_config = config["general"]
        states_config = config["state_variables"]
        actions_config = config["action_variables"]
        parameters_config = config["parameters"]

        self.num_batches = general_config["num_batches"]
        self.state_config = states_config

        self.equations_of_motion = Equations_of_motion_Ace_Dice_2016(
            general_config["t_max"], states_config, actions_config, parameters_config
        )
        self.loss = Computeloss_Ace_Dice_2016(
            parameters_config, self.equations_of_motion
        )

    def step(self, batch_s_t: tf.Tensor, batch_a_t: tf.Tensor) -> tf.Tensor:
        """
        Transitions the agent from current state s_t to next state s_{t+1} by taking action a_t for all batches.

        Args:
            batch_s_t: tf.Tensor of dimensions [batch, state, state variables]
            batch_a_t: tf.Tensor of dimensions [batch, actions, action variables]

        Returns:
            batch_s_{t+1}: tf.Tensor of dimensions [batch, state, state variables]
        """
        next_states = []
        for s_t, a_t in zip(batch_s_t, batch_a_t):
            s_tplus = self.equations_of_motion.update_state(s_t, a_t)
            s_tplus_casted = tf.cast(s_tplus, tf.float32)
            next_states.append(s_tplus_casted)

        next_states_tensor = tf.stack(next_states)

        return tf.cast(next_states_tensor, dtype=tf.float32)

    def compute_loss(
        self,
        batch_s_t: tf.Tensor,
        batch_a_t: tf.Tensor,
        batch_s_tplus: tf.Tensor,
        batch_a_tplus: tf.Tensor,
    ) -> float:
        """_summary_

        Args:
            batch_s_t (tf.Tensor): _description_
            batch_a_t (tf.Tensor): _description_
            batch_s_tplus (tf.Tensor): _description_
            batch_a_tplus (tf.Tensor): _description_

        Returns:
            float: (mse without penalty, mse with penalty)
        """
        total_mse = tf.constant(0.0, dtype=tf.float32)
        sample_counter = tf.constant(0, dtype=tf.float32)

        for s_t, a_t, st_plus, a_tplus in zip(
            batch_s_t, batch_a_t, batch_s_tplus, batch_a_tplus
        ):
            se_no_penalty = self.loss.squared_error_for_transition(
                s_t, a_t, st_plus, a_tplus
            )
            total_mse += se_no_penalty
            sample_counter += 1.0

        total_mse = total_mse / sample_counter
        total_mse = tf.cast(total_mse, dtype=tf.float32)

        return total_mse

    def reset(self) -> tf.Tensor:
        """
        Resets state to the initial states in the config file.
        Returns initial states.
        """
        batches = []
        for batch in range(self.num_batches):
            state = []
            for init_val in self.state_config:
                val = tf.constant(init_val.get("init_val"), dtype=tf.float32)
                if init_val.get("name") == "k_t":
                    val = tf.math.log(val)
                state.append(val)
            state_tensor = tf.convert_to_tensor(state)
            batches.append(state_tensor)

        state_tensor = tf.stack(batches)
        return tf.cast(state_tensor, dtype=tf.float32)
