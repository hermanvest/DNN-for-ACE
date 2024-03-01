import numpy as np
import tensorflow as tf

from typing import Tuple
from agents.deqn_agent import DEQN_agent
from environments.deqn_ace_dice.ace_dice_env import Ace_dice_env


class Algorithm_DEQN:
    def __init__(
        self,
        n_iterations: int,
        n_epochs: int,
        t_max: int,
        batch_size: int,
        env: Ace_dice_env,
        agent: DEQN_agent,
        optimizer: tf.keras.optimizers.Optimizer,
        log_dir: str = "logs/train",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        # Initializations related to the algorithm
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.t_max = t_max

        # Adjusting task difficulty as network gets better
        self.thresholds_and_sizes = [
            (500, 32),  # "moderate loss threshold"
            (200, 64),  # "small loss threshold"
        ]

        self.episodelengths = [
            4,  # sorted easiest to hardest
            8,
            16,
            32,
            64,
            128,
            256,
            300,
        ]

        # Initializations for env, agent and optimizer
        self.env = env
        self.agent = agent
        self.optimizer = optimizer

        # Initializations related to logging model performance and model checkpointing
        self.writer = tf.summary.create_file_writer(log_dir)
        self.checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, model=self.agent.policy_network
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=checkpoint_dir, max_to_keep=5
        )

        # Restore the latest checkpoint
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(
                f"\nRestoring policynetwork from {self.checkpoint_manager.latest_checkpoint}"
            )
        else:
            print("\nInitializing policy network from scratch.")

    ################ HELPER FUNCITONS ################
    def print_time_elapsed(
        self, current_time: tf.Tensor, start_time: tf.Tensor
    ) -> None:
        """
        Converts a duration from seconds to a formatted string showing the equivalent time in hours, minutes, and seconds.

        Args:
            current_time (tf.Tensor): A tensor representing the current time in seconds.
            start_time (tf.Tensor): A tensor representing the start time in seconds.

        Returns:
            str: A string representing the elapsed time in the format "Hh Mm Ss", where H, M, and S are hours, minutes, and seconds, respectively.
        """
        elapsed_time = current_time - start_time
        elapsed_time = float(elapsed_time.numpy())

        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60
        seconds = (elapsed_time % 3600) % 60

        print(
            f"========== Time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s =========="
        )

    def choose_batch_size(self, start: bool, total_loss: tf.Tensor = None) -> int:
        # Handling the start condition by setting the batch size to the smallest one defined
        if start:
            # Assuming thresholds_and_sizes are sorted in descending order of thresholds
            # The smallest batch size is the second element of the last tuple in the list
            batch_size = self.thresholds_and_sizes[0][1]
            return batch_size

        # Ensure total_loss is a scalar and get its value
        batch_size = 128

        if total_loss is not None:
            total_loss_value = total_loss.numpy()

            # Reverse iterate to pop smaller thresholds without affecting indices of unvisited items
            for i in reversed(range(len(self.thresholds_and_sizes))):
                threshold, batch_size_i = self.thresholds_and_sizes[i]

                # If total_loss is greater than the current threshold, update batch_size
                if total_loss_value >= threshold:
                    batch_size = batch_size_i

                # If total_loss is less than the current threshold, pop this threshold off the list
                else:
                    self.thresholds_and_sizes.pop(i)
        return batch_size

    def choose_episode_length(self, total_loss: tf.Tensor = None) -> None:
        if len(self.episodelengths) == 1:
            return
        if total_loss.numpy() < 20:
            self.episodelengths.pop(0)

    ################ MAIN TRAINING FUNCTIONS ################
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

        for _ in range(self.t_max):
            a_t = self.agent.get_actions(s_t)  # shape [batch, actionvars]
            s_tplus = self.env.step(s_t, a_t)
            states.append(s_t)
            s_t = s_tplus

        states_tensor = tf.stack(states)
        return tf.cast(states_tensor, dtype=tf.float32)

    def epoch(self, batches: tf.Tensor, step_index: int) -> tf.Tensor:
        """
        Processes a series of batches for a single epoch, computes the loss for each batch,
        applies gradients, and returns the average loss across all batches as a float32 tensor.

        Args:
            batches (tf.Tensor): A batched dataset of inputs for the epoch.
            step_index (int): episode number*epoch number

        Returns:
            tf.Tensor: The mean loss across all batches for the epoch as a float32 tensor.
        """
        total_epoch_loss = tf.constant(0.0, dtype=tf.float32)
        num_batches = tf.constant(0, dtype=tf.float32)

        for batch_index, batch_s_t in enumerate(batches):
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

            total_epoch_loss += loss
            num_batches += 1.0

            if any([tf.reduce_any(tf.math.is_nan(grad)) for grad in gradients]):
                print("NaN gradient encountered for loss")

            ###### Logging          ######
            global_norm = tf.linalg.global_norm(gradients)
            with self.writer.as_default():
                tf.summary.scalar(
                    "Gradient Norm",
                    global_norm,
                    step=step_index * len(batches) + batch_index,
                )
                self.writer.flush()

        # MSE over all batches
        epoch_mse = total_epoch_loss / num_batches
        epoch_mse = tf.cast(epoch_mse, dtype=tf.float32)

        return epoch_mse

    def train_on_episodes(
        self, episodes: tf.Tensor, iteration_number: int
    ) -> tf.Tensor:
        """
        Trains the model on a dataset of episodes, reshaping and shuffling the episodes
        before processing them in batches for each epoch. The method computes the mean
        squared error loss for each epoch and returns the average loss across all epochs.

        The episodes tensor is first flattened to combine the batch and timestep dimensions,
        which is then shuffled to ensure variability in the training batches. For each epoch,
        the shuffled episodes are divided into batches based on the predefined batch size,
        and the model is trained on these batches. The episodes are reshuffled after each epoch.

        Args:
            episodes (tf.Tensor): A 3D tensor of shape [batch_numbers, timesteps, state_variables]
                                representing the collected episodes to train on.
            iteration_number (int): Integer representing current iteration in the main loop

        Returns:
            tf.Tensor: The mean squared error loss averaged over all epochs, as a float32 tensor.
        """
        # Flattened, means shape from [batchnumbers, timesteps, state_variables] to [batchnumbers*timesteps, state_variables]
        flattened_episodes = tf.reshape(episodes, [-1, episodes.shape[-1]])
        shuffled_episodes = tf.random.shuffle(flattened_episodes)

        total_epochs_loss = tf.constant(0.0, dtype=tf.float32)
        num_batches = tf.constant(0, dtype=tf.float32)

        for epoch_i in range(self.n_epochs):
            batches = tf.data.Dataset.from_tensor_slices(shuffled_episodes).batch(
                self.batch_size
            )
            # Calculating losses
            epoch_mse = self.epoch(batches, iteration_number)
            total_epochs_loss += epoch_mse
            num_batches += 1.0

            # Shuffling up the episodes for the next iteration
            shuffled_episodes = tf.random.shuffle(shuffled_episodes)

            print(f"Epoch {epoch_i+1}/{self.n_epochs}: Total Loss = {epoch_mse}")

        # MSE over all batches
        mean_epoch_loss = total_epochs_loss / num_batches
        mean_epoch_loss = tf.cast(mean_epoch_loss, dtype=tf.float32)

        return mean_epoch_loss

    def main_loop(self) -> None:
        training_start = tf.timestamp()
        self.batch_size = self.choose_batch_size(True)

        for iteration_i in range(self.n_iterations):
            print(f"\nStarting iteration {iteration_i+1}/{self.n_iterations}")

            ###### Algorithm steps  ######
            episode = self.generate_episodes()
            iteration_loss = self.train_on_episodes(episode, iteration_i)

            # self.batch_size = self.choose_batch_size(False, iteration_loss)
            # self.choose_episode_length(iteration_loss)

            ###### Logging          ######
            print(
                f"Loss for iteration {iteration_i+1}/{self.n_iterations}: Loss = {iteration_loss}"
            )

            with self.writer.as_default():
                tf.summary.scalar(
                    "Mean squared errors averaged over all epochs",
                    iteration_loss,
                    step=iteration_i,
                )
                self.writer.flush()

            # Save the model checkpoint
            self.checkpoint_manager.save()
            print(f"Checkpoint saved at {self.checkpoint_manager.latest_checkpoint}")

            self.print_time_elapsed(tf.timestamp(), training_start)
