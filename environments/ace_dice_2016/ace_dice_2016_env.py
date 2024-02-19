import tensorflow as tf
import numpy as np

from typing import Any, Dict, Tuple, List
from environments.abstract_environment import Abstract_Environment
from equations_of_motion import Equations_of_motion_Ace_Dice
from compute_loss import Computeloss


class Ace_dice_2016(Abstract_Environment):
    def __init__(
        self,
        num_batches: int,
        state_config: List,
        parameter_config: Dict[str, Any],
    ) -> None:
        self.num_batches = num_batches
        self.state_config = state_config
        self.equations_of_motion = Equations_of_motion_Ace_Dice(
            state_config, parameter_config
        )
        self.loss = Computeloss()

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
            next_states.append(s_tplus)

        next_states_tensor = tf.stack(next_states)
        return next_states_tensor

    def compute_loss(
        self, batch_s_t: tf.Tensor, batch_a_t: tf.Tensor
    ) -> Tuple[float, float]:
        """_summary_

        Args:
            batch_s_t (tf.Tensor): _description_
            batch_a_t (tf.Tensor): _description_

        Returns:
            Tuple[float, float]: (mse without penalty, mse with penalty)
        """
        # TODO: just input s_t, a_t, s_tplus, a_tplus!
        # get the next state taken from current step
        s_tplus_batch = self.step(batch_s_t, batch_a_t)

        total_mse = 0.0
        total_mse_with_penalty = 0.0

        for s_t, a_t, s_t_plus in zip(batch_s_t, batch_a_t, s_tplus_batch):
            se_no_penalty, se_penalty = self.loss.squared_error_for_transition(
                s_t, a_t, s_t_plus
            )
            total_mse += se_no_penalty
            total_mse_with_penalty += se_penalty

        return total_mse, total_mse_with_penalty

    def reset(self) -> tf.Tensor:
        """
        Resets state to the initial states in the config file.
        Returns initial states.
        """
        batches = []
        for batch in range(self.num_batches):
            state = []
            for init_val in self.state_config:
                state.append(init_val.get("init_val"))
            batches.append(tf.convert_to_tensor(state))

        state_tensor = tf.stack(state)
        return state_tensor
