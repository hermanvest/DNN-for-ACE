import numpy as np
import tensorflow as tf
from typing import Tuple
from agents.deqn_agent import DEQN_agent
from environments.ace_dice_2016.ace_dice_2016_env import Ace_dice_2016


class Algorithm_DEQN:
    def __init__(
        self,
        n_episodes: int,
        n_epochs: int,
        t_max: int,
        batch_size: int,
        env: Ace_dice_2016,
        agent: DEQN_agent,
        optimizer: tf.keras.optimizers.Optimizer,
        log_dir: str = "logs/train",
    ) -> None:
        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.t_max = t_max
        self.batch_size = batch_size
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.writer = tf.summary.create_file_writer(log_dir)

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
        return tf.cast(states_tensor, dtype=tf.float32)

    def epoch(self, batches: tf.Tensor) -> tf.Tensor:
        """_summary_

        Args:
            batches (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        total_loss = 0.0
        num_batches = 0

        for batch_s_t in batches:
            with tf.GradientTape() as tape:
                batch_a_t = self.agent.get_actions(batch_s_t)
                batch_s_tplus = self.env.step(batch_s_t, batch_a_t)
                batch_a_tplus = self.agent.get_actions(batch_s_tplus)

                loss = self.env.compute_loss(
                    batch_s_t, batch_a_t, batch_s_tplus, batch_a_tplus
                )

            gradients = tape.gradient(
                loss, self.agent.policy_network.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients, self.agent.policy_network.trainable_variables)
            )

            total_loss += loss
            num_batches += 1

        # Average MSE over all batches
        average_mse = total_loss / num_batches

        return average_mse

    def train_on_episodes(self, episodes: tf.Tensor) -> None:
        """_summary_

        Args:
            episodes (tf.Tensor): _description_
        """
        # Flattened, means shape from [batchnumbers, timesteps, state_variables] to [batchnumbers*timesteps, state_variables]
        flattened_episodes = tf.reshape(episodes, [-1, episodes.shape[-1]])
        shuffled_episodes = tf.random.shuffle(flattened_episodes)

        for epoch_i in self.n_epochs:
            batches = tf.data.Dataset.from_tensor_slices(shuffled_episodes).batch(
                self.batch_size
            )
            epoch_average_mse = self.epoch(batches)
            shuffled_episodes = tf.random.shuffle(shuffled_episodes)

            # Logging
            with self.writer.as_default():
                tf.summary.scalar("Total Epoch Loss", epoch_average_mse, step=epoch_i)
                self.writer.flush()
            print(
                f"Epoch {epoch_i+1}/{self.n_epochs}: Total Loss = {epoch_average_mse}"
            )

    def main_loop(self) -> None:
        training_start = tf.timestamp()

        for episode_i in range(self.n_episodes):
            print(f"Starting Episode {episode_i+1}/{self.n_episodes}")
            episode = self.generate_episodes()
            self.train_on_episodes(episode)

            print(f"==== Time elapsed: {tf.timestamp()-training_start} ====")

        # TODO: checkpoint of model if it is performing better
