from typing import Any, Dict
from environments.abstract_environment import Abstract_Environment
import tensorflow as tf
import numpy as np


class Ace_dice_2016(Abstract_Environment):
    def __init__(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    def update_state(self, s_t: tf.Tensor, a_t: tf.Tensor) -> tf.Tensor:
        # dumbest way to do this:
        pass

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
            s_tplus = self.update_state(s_t, a_t)
            next_states.append(s_tplus)

        next_states_tensor = tf.stack(next_states)
        return next_states_tensor

    def compute_loss(self, batch: np.ndarray) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets state to the initial states in the config file.
        Returns initial states.
        """
        raise NotImplementedError
