import numpy as np
import tensorflow as tf
from typing import Tuple

from tensorflow.keras.optimizers import Optimizer
from agents.deqn_agent import DEQN_agent
from environments.abstract_environment import Abstract_Environment


class Algorithm_DEQN:
    def __init__(
        self,
        n_episodes: int,
        n_epochs: int,
        t_max: int,
        batch_size: int,
        env: Abstract_Environment,
        agent: DEQN_agent,
        optimizer: Optimizer,
    ) -> None:
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.t_max = t_max
        self.batch_size = batch_size
        self.env = env
        self.agent = agent
        self.optimizer = optimizer

    def check_compatibility(self) -> None:
        """
        Chekcs if the agent's inputspace is the same as the state space of the environment.
        checks if the agent's outputspace is compatible with the actionspace in the environment.

        Raises:
            ValueError: inputs are not compatible
        """
        # raise ValueError(f"Incompatible parameters: {param1} and {param2}")
        raise NotImplementedError

    def generate_episodes(self) -> tf.Tensor:
        """
        Generates an episode with the policy network and returns a tensor with states visited.
        Args:
            None

        Returns:
            tf.Tensor: [timestep, batches, statevariables]
        """

        s_t = self.env.reset()  # shape [batch, statevariables]
        states = []  # shape [timestep, batchnumber, statevariables]

        for t in range(self.t_max):
            a_t = self.agent.get_actions(s_t)  # shape [batch, actionvars]
            s_tplus = self.env.step(s_t, a_t)
            states.append(s_t)
            s_t = s_tplus
        states_tensor = tf.stack(states)
        return states_tensor

    def epoch(self, batches: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: Consider renaming batches to statesample. More informative?
        total_loss = 0.0
        total_loss_with_penalty = 0.0
        num_batches = 0

        for batch in batches:
            with tf.GradientTape() as tape:
                # TODO: make sure prediction is with old parameters?????
                a_t = self.agent.get_action(batch)
                loss, loss_with_penalty = self.env.compute_loss(batch, a_t)

            gradients = tape.gradient(
                loss, self.agent.policy_network.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.agent.policy_network.trainable_variables)
            )

            total_loss += loss
            total_loss_with_penalty += loss_with_penalty
            num_batches += 1

        # Average MSE over all batches
        average_mse = total_loss / num_batches
        average_mse_with_penalty = total_loss_with_penalty / num_batches

        return average_mse, average_mse_with_penalty

    def train_on_episodes(self, episodes: tf.Tensor) -> None:
        # Flattened, means shape from [batchnumbers, timesteps, state_variables] to [batchnumbers*timesteps, state_variables]
        flattened_episodes = tf.reshape(episodes, [-1, episodes.shape[-1]])
        shuffled_episodes = tf.random.shuffle(flattened_episodes)

        for epoch_i in self.n_epochs:
            batches = tf.data.Dataset.from_tensor_slices(shuffled_episodes).batch(
                self.batch_size
            )
            total_loss, loss_without_penalty_bounds = self.epoch(batches)
            shuffled_episodes = tf.random.shuffle(shuffled_episodes)
            # TODO: Log loss with and without bound penalization

    def main_loop(self) -> None:
        training_start = tf.timestamp()

        for episode_i in range(self.n_episodes):
            episode = self.generate_episodes()
            self.train_on_episodes(episode)

            print(f"==== Time elapsed: {tf.timestamp()-training_start} ====")

        # TODO: checkpoint of model if it is performing better
        # TODO: create relevant plots for the best performing model
