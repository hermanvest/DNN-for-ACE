import tensorflow as tf
from typing import Any, Dict


class DEQN_agent:
    """
    __summary

    Attributes:
        policy_network (tf.keras.Model): A TensorFlow Keras model representing the policy network of the agent. This network takes the current state of the environment as input and outputs a decision or action to be taken by the agent.
    """

    def __init__(self, policy_network: tf.keras.Model) -> None:
        self.policy_network = policy_network

    def get_actions(self, state: tf.Tensor) -> tf.Tensor:
        """Returns actions for given state as per current policy."""
        return self.policy_network(state)

    # TODO: Implement save
    def save(self, filename):
        """Save the agent's model parameters."""
        # Implement model saving logic
        raise NotImplementedError

    # TODO: Implement load
    def load(self, filename):
        """Load the agent's model parameters."""
        # Implement model loading logic
        raise NotImplementedError
