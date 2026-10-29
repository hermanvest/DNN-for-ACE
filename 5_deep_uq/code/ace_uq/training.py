"""DEQN training with parameters as pseudo-states.

As in part 2, each iteration simulates episodes with the current policy and runs a
few epochs of mini-batch gradient descent on the summed squared residuals. The
difference is that every iteration simulates several trajectories in parallel,
each with its own parameter draw from the training box, so the visited states
cover the parameter space as well as the economic states.

Optionally, the recursive conditions are weighted by 1 / (1 - beta) of each
sample (see `discount_weights`).

The weights kept are those with the lowest objective on a fixed set of
validation draws, the counterpart of part 2's lowest episode loss.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf

from ace_uq.model import AceDiceUQ
from ace_uq.residuals import residuals


class Trainer:
    def __init__(self, model: AceDiceUQ, network: tf.keras.Model, config: Dict, out_dir: str, seed: int = 1):
        self.model = model
        self.network = network
        self.config = config["training"]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        states = self.config["trajectories"] * model.t_max
        self.n_batches = int(np.ceil(states / self.config["batch_size"]))
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.config["learning_rate"],
            decay_steps=self.config["iterations"] * self.config["epochs"] * self.n_batches,
            alpha=self.config["final_learning_rate"] / self.config["learning_rate"],
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=self.config["clipnorm"])
        self.validation = model.sample_parameters(self.config["validation_draws"], np.random.default_rng(12345))

    def policy(self, s: tf.Tensor) -> Dict[str, tf.Tensor]:
        return self.model.decode_actions(s, self.network(self.model.preprocess(s)))

    @tf.function
    def _simulate_step(self, s: tf.Tensor) -> tf.Tensor:
        return self.model.step(s, self.policy(s))

    def simulate(self, parameters: np.ndarray, n: int = None) -> tf.Tensor:
        """Trajectories under the current policy, shape [draws, n, 10]."""
        s = self.model.initial_state(parameters)
        states = []
        for _ in range(n or self.model.t_max):
            states.append(s)
            s = self._simulate_step(s)
        return tf.stack(states, axis=1)

    def objective(self, s: tf.Tensor) -> tf.Tensor:
        a = self.policy(s)
        s_next = self.model.step(s, a)
        a_next = self.policy(s_next)
        r = residuals(self.model, s, a, s_next, a_next)
        if self.config["discount_weighting"]:
            r *= self.discount_weights(s)
        return tf.reduce_mean(tf.reduce_sum(tf.square(r), axis=1))

    def discount_weights(self, s: tf.Tensor) -> tf.Tensor:
        """Weights 1 / (1 - beta) on the recursive conditions, relative to the baseline.

        A residual eps in a recursion such as lambda = beta (... + lambda') becomes an
        error of order eps / (1 - beta) in lambda, so a low discount rate needs smaller
        residuals for the same accuracy. Weighting the recursive conditions (capital,
        carbon and temperature costates, Bellman) by 1 / (1 - beta) evens this out.
        The static first-order and KKT conditions keep weight one. Every weight is
        positive, so the optimum does not move.
        """
        beta_baseline = (1.0 + self.model.baseline[0]) ** -self.model.timestep
        w = (1.0 - beta_baseline) / (1.0 - self.model.beta(s))
        ones = tf.ones_like(w)
        # Residual order: l0, l1 (3), l2 (2), l3, l4, l5A, l5B, l6
        return tf.stack([w, w, w, w, w, w, ones, ones, ones, ones, w], axis=1)

    @tf.function
    def _train_step(self, s: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.objective(s)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        return loss

    @tf.function
    def _objective(self, s: tf.Tensor) -> tf.Tensor:
        return self.objective(s)

    def validation_loss(self) -> float:
        states = self.simulate(self.validation)
        return float(self._objective(tf.reshape(states, (-1, states.shape[-1]))))

    def train(self) -> Dict[str, list]:
        cfg = self.config
        history = {"iteration": [], "loss": [], "seconds": [], "validation": []}
        start = time.time()
        best = np.inf
        for iteration in range(cfg["iterations"]):
            parameters = self.model.sample_parameters(cfg["trajectories"], self.rng)
            states = self.simulate(parameters)
            states = tf.reshape(states, (-1, states.shape[-1]))
            losses = []
            for _ in range(cfg["epochs"]):
                shuffled = tf.random.shuffle(states)
                for batch in tf.split(shuffled, self.n_batches):
                    losses.append(float(self._train_step(batch)))

            history["iteration"].append(iteration)
            history["loss"].append(float(np.mean(losses)))
            history["seconds"].append(time.time() - start)

            last = iteration == cfg["iterations"] - 1
            if iteration % cfg["validate_every"] == 0 or last:
                validation = self.validation_loss()
                history["validation"].append({"iteration": iteration, "loss": validation})
                if validation < best:
                    best = validation
                    self.network.save_weights(os.path.join(self.out_dir, "policy.weights.h5"))
                print(
                    f"iter {iteration:5d}  loss {history['loss'][-1]:.3e}  validation {validation:.3e}  "
                    f"best {best:.3e}  {history['seconds'][-1] / 60:.1f} min",
                    flush=True,
                )
                with open(os.path.join(self.out_dir, "history.json"), "w") as f:
                    json.dump(history, f)

        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        return history
