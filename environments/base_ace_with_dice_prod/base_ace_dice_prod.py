from typing import Any, Dict
from ..abstract_environment import Abstract_Environment
import tensorflow as tf


class Base_ace_dice_prod(Abstract_Environment):
    metadata = {"render_modes": ["none", "human"]}

    def __init__(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _get_obs(self) -> None:
        raise NotImplementedError

    def _update_state(self) -> None:
        raise NotImplementedError

    def step(self, policy_output: tf.Tensor) -> tf.Tensor:
        """
        Takes as input, the action obtained from the policy.
        Returns the state we trantision to by taking the step in the environment.
        """
        raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets state to the initial states in the config file.
        Returns initial states.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Terminates environment
        """
        raise NotImplementedError
