"""DEQN training loop (thesis Algorithm 1).

Each iteration simulates one episode with the current policy, then runs a few
epochs of mini-batch gradient descent on the policy objective evaluated on the
visited states.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf

from ace_deqn.model import AceDice
from ace_deqn.residuals import residuals


class Trainer:
    def __init__(self, model: AceDice, network: tf.keras.Model, config: Dict, out_dir: str):
        self.model = model
        self.network = network
        self.config = config["training"]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        steps_per_iteration = self.config["epochs"] * int(
            np.ceil(model.t_max / self.config["batch_size"])
        )
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.config["learning_rate"],
            decay_steps=self.config["iterations"] * steps_per_iteration,
            alpha=self.config["final_learning_rate"] / self.config["learning_rate"],
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=schedule, clipnorm=self.config["clipnorm"]
        )

    def policy(self, s: tf.Tensor) -> Dict[str, tf.Tensor]:
        return self.model.decode_actions(s, self.network(self.model.preprocess(s)))

    @tf.function
    def _simulate_step(self, s: tf.Tensor) -> tf.Tensor:
        return self.model.step(s, self.policy(s))

    def simulate_episode(self) -> tf.Tensor:
        s = self.model.initial_state(1)
        states = []
        for _ in range(self.model.t_max):
            states.append(s)
            s = self._simulate_step(s)
        return tf.concat(states, axis=0)

    def objective(self, s: tf.Tensor) -> tf.Tensor:
        """Policy objective (eq. 3.10): mean over the batch of the summed squared residuals."""
        a = self.policy(s)
        s_next = self.model.step(s, a)
        a_next = self.policy(s_next)
        r = residuals(self.model, s, a, s_next, a_next)
        return tf.reduce_mean(tf.reduce_sum(tf.square(r), axis=1))

    @tf.function
    def _train_step(self, s: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.objective(s)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    def train(self) -> Dict[str, list]:
        history = {"iteration": [], "loss": [], "max_batch_loss": [], "seconds": []}
        start = time.time()
        best = np.inf
        for iteration in range(self.config["iterations"]):
            episode = self.simulate_episode()
            losses = []
            for _ in range(self.config["epochs"]):
                shuffled = tf.random.shuffle(episode)
                for batch in tf.split(shuffled, int(np.ceil(self.model.t_max / self.config["batch_size"]))):
                    losses.append(float(self._train_step(batch)))

            history["iteration"].append(iteration)
            history["loss"].append(float(np.mean(losses)))
            history["max_batch_loss"].append(float(np.max(losses)))
            history["seconds"].append(time.time() - start)

            # Keep the weights with the lowest objective on the full episode
            episode_loss = float(self.objective(episode))
            if episode_loss < best:
                best = episode_loss
                self.network.save_weights(os.path.join(self.out_dir, "policy.weights.h5"))

            if iteration % 50 == 0 or iteration == self.config["iterations"] - 1:
                print(
                    f"iter {iteration:5d}  loss {history['loss'][-1]:.3e}  "
                    f"episode {episode_loss:.3e}  best {best:.3e}  "
                    f"{history['seconds'][-1] / 60:.1f} min",
                    flush=True,
                )
                with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                    json.dump(history, f)

        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        return history
