import numpy as np
import tensorflow as tf
from typing import Tuple

from agents.deqn_agent import DEQN_agent
from environments.abstract_environment import Abstract_Environment


class Algorithm_DEQN:
    def __init__(
        self,
        n_episodes: int,
        n_epochs: int,
        t_max: int,
        env: Abstract_Environment,
        agent: DEQN_agent,
    ) -> None:
        self.n_episodes = n_episodes

        self.env = env
        self.agent = agent

    def check_compatibility(self) -> None:
        """
        Chekcs if the agent's inputspace is the same as the state space of the environment.
        checks if the agent's outputspace is compatible with the actionspace in the environment.

        Raises:
            ValueError: inputs are not compatible
        """
        # raise ValueError(f"Incompatible parameters: {param1} and {param2}")
        raise NotImplementedError

    def generate_episode(self) -> tf.Tensor:
        """
        Generates an episode with the policy network and returns a tensor with states visited.
        Args:
            None

        Returns:
            tf.Tensor: [states]
        """
        # TODO: Extend so that we can perform multiple batches at the same time

        s_t = self.env.reset()
        states = []

        for t in range(self.t_max):
            a_t = self.agent.get_action(s_t)
            s_tplus = self.env.step(**a_t)

            s_t_tensor = tf.convert_to_tensor(s_t, dtype=tf.float32)
            states.append(s_t_tensor)

            s_t = s_tplus
        states_tensor = tf.stack(states)
        return states_tensor

    def epoch(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError
        # 1. divide into batches: get a list of batches with shape [minibatchnumber, content of minibatch]
        # 2. for batch in batches
        #   2.1 calculate errors - for some reason, they do new predictions on the states in the episode?
        #   Seems like they use parameters from the first run to continue the updates? Inefficient!
        #   2.2 do gradient descent
        # 3. return losses

    def train_on_episode(self, episode: tf.Tensor) -> None:
        # TODO: create batches from the tensor
        for epoch_i in self.n_epochs:
            total_loss, loss_without_penalty_bounds = self.epoch()
            # TODO: Log loss with and without bound penalization

    def main_loop(self) -> None:
        training_start = tf.timestamp()

        for episode_i in range(self.n_episodes):
            episode = self.generate_episode()
            self.train_on_episode(episode)

            print(f"==== Time elapsed: {tf.timestamp()-training_start} ====")

        # TODO: checkpoint of model if it is performing better
        # TODO: create relevant plots for the best performing model
