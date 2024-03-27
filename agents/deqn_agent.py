import tensorflow as tf
from typing import Any, Dict


class DEQN_agent:
    """summary"""

    def __init__(self, policy_network: tf.keras.Model) -> None:
        self.policy_network = policy_network

    def get_actions(self, state: tf.Tensor) -> tf.Tensor:
        """Returns actions for given state as per current policy."""

        return self.policy_network(self.preprocess(state))

    def preprocess(
        self,
        state: tf.Tensor,
        varsigma: tf.Tensor = tf.constant(0.02, dtype=tf.float32),
    ) -> tf.Tensor:
        """Applies time transformation as in Traeger (2014) and downscales carbon reservoirs from GtCO2 to 1000GtCO2.

        Time scaled as: hat(t) = -exp(-varsigma t)

        Args:
            state (tf.Tensor): Shape [batch, state, state variables]

        Returns:
            preprocessed states (tf.Tensor): Shape [batch, state, state variables]
        """
        # Time located at index 6, where we select index 6 to, but not including 7.
        t = state[..., 6:7]
        hat_t = -tf.exp(-varsigma * t) + 1

        m = state[..., 1:4]
        m_scaled = m / 1000

        preprocessed_state = tf.concat(
            [state[..., :1], m_scaled, state[..., 4:6], hat_t], axis=-1
        )

        return preprocessed_state
