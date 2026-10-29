"""Group Relative Policy Optimization (GRPO) for ACE.

GRPO is PPO without a critic: for each prompt, a group of responses is sampled and each
response's advantage is its reward normalized within the group. Here a prompt is a start
state, a response is a trajectory of `horizon` periods from it, and the policy is a
Gaussian in logit space over the consumption rate x and the abatement rate mu:

    z ~ N(m_theta(s), diag(sigma^2)),   x = sigmoid(z_1),   mu = sigmoid(z_2).

Advantages are either per-step ("process"): the discounted return-to-go of each action,
normalized across the group at the same step; or per-trajectory ("outcome"): the
discounted return of the whole trajectory, normalized across the group and shared by
all of its actions. The policy is updated with the PPO clipped objective. There is no
KL penalty, as there is no reference policy.

Nothing is differentiated through the model: the environment is only sampled, as in
model-free reinforcement learning.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf

from ace_grpo.env import EPSILON, AceEnv

LOG_2PI = float(np.log(2 * np.pi))


def squash(z: tf.Tensor) -> tf.Tensor:
    return EPSILON + (1 - 2 * EPSILON) * tf.sigmoid(z)


class GRPO:
    def __init__(self, env: AceEnv, config: Dict) -> None:
        self.env = env
        self.cfg = config["grpo"]
        policy = config["policy"]

        initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=policy["seed"]
        )
        inputs = tf.keras.Input(shape=(7,))
        h = inputs
        for _ in range(policy["hidden_layers"]):
            h = tf.keras.layers.Dense(policy["hidden_nodes"], activation=policy["activation"], kernel_initializer=initializer)(h)
        # Near-zero output layer, as is standard for PPO policies: the initial policy has
        # mean x = mu = 0.5 in every state, so exploration is not lost to a saturated sigmoid
        output_init = tf.keras.initializers.VarianceScaling(scale=1e-4, mode="fan_avg", distribution="uniform", seed=policy["seed"])
        self.network = tf.keras.Model(inputs, tf.keras.layers.Dense(2, kernel_initializer=output_init)(h))
        self.log_std = tf.Variable(tf.fill([2], float(policy["initial_log_std"])), name="log_std")
        self.min_log_std = float(policy["min_log_std"])
        self.variables = self.network.trainable_variables + [self.log_std]

        samples = self.cfg["prompts"] * self.cfg["group_size"] * self.cfg["horizon"]
        steps = self.cfg["iterations"] * self.cfg["update_epochs"] * self.cfg["minibatches"]
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            self.cfg["learning_rate"], decay_steps=steps, alpha=self.cfg["final_learning_rate_factor"]
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=self.cfg["clipnorm"])
        self.samples_per_iteration = samples

    ################ POLICY ################
    def mean_logits(self, s: tf.Tensor) -> tf.Tensor:
        return self.network(self.env.preprocess(s))

    def std(self) -> tf.Tensor:
        return tf.exp(tf.maximum(self.log_std, self.min_log_std))

    def log_prob(self, s: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        log_std = tf.maximum(self.log_std, self.min_log_std)
        u = (z - self.mean_logits(s)) / tf.exp(log_std)
        return tf.reduce_sum(-0.5 * tf.square(u) - log_std - 0.5 * LOG_2PI, axis=-1)

    def policy(self, s: tf.Tensor) -> Dict[str, tf.Tensor]:
        """The deterministic policy (mean action), used for evaluation."""
        z = self.mean_logits(s)
        return {"x": squash(z[:, 0]), "mu": squash(z[:, 1])}

    ################ ROLLOUTS ################
    @tf.function
    def _deterministic_step(self, s):
        return self.env.step(s, self.policy(s))

    def simulate(self, n: int) -> tf.Tensor:
        s = self.env.initial_state(1)
        states = []
        for _ in range(n):
            states.append(s)
            s = self._deterministic_step(s)
        return tf.concat(states, axis=0)

    @tf.function
    def _sample_step(self, s):
        z = self.mean_logits(s) + self.std() * tf.random.normal((tf.shape(s)[0], 2))
        a = {"x": squash(z[:, 0]), "mu": squash(z[:, 1])}
        return z, self.log_prob(s, z), self.env.utility(s, a), self.env.step(s, a)

    def rollout(self, prompts: tf.Tensor):
        """Roll out a group of trajectories from every prompt; returns tensors of shape [B, H, ...]."""
        s = tf.repeat(prompts, self.cfg["group_size"], axis=0)
        states, actions, log_probs, rewards = [], [], [], []
        for _ in range(self.cfg["horizon"]):
            z, lp, r, s_next = self._sample_step(s)
            states.append(s)
            actions.append(z)
            log_probs.append(lp)
            rewards.append(r)
            s = s_next
        return tuple(tf.stack(x, axis=1) for x in (states, actions, log_probs, rewards))

    def advantages(self, rewards: tf.Tensor) -> tf.Tensor:
        """Group-relative advantages, shape [B, H]."""
        P, G, H = self.cfg["prompts"], self.cfg["group_size"], self.cfg["horizon"]
        returns = np.zeros(rewards.shape, dtype=np.float32)
        running = np.zeros(rewards.shape[0], dtype=np.float32)
        r = rewards.numpy()
        for h in reversed(range(H)):
            running = r[:, h] + self.env.beta * running
            returns[:, h] = running
        if self.cfg["advantage"] == "outcome":
            returns = np.repeat(returns[:, :1], H, axis=1)
        grouped = returns.reshape(P, G, H)
        normalized = (grouped - grouped.mean(axis=1, keepdims=True)) / (grouped.std(axis=1, keepdims=True) + 1e-8)
        return tf.constant(normalized.reshape(P * G, H))

    ################ UPDATE ################
    @tf.function
    def _update(self, s, z, old_log_prob, advantage):
        with tf.GradientTape() as tape:
            ratio = tf.exp(self.log_prob(s, z) - old_log_prob)
            clipped = tf.clip_by_value(ratio, 1 - self.cfg["clip"], 1 + self.cfg["clip"])
            loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped * advantage))
        grads = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables))
        return loss

    ################ EVALUATION ################
    def quick_eval(self) -> Dict[str, float]:
        horizon = 58  # periods up to about 2300
        s = self.simulate(horizon)
        a = self.policy(s)
        t = s[:, 6].numpy().astype(int)
        return {
            "x_err": float(np.max(np.abs(a["x"].numpy() - self.env.costates.x))),
            "mu_err": float(np.mean(np.abs(a["mu"].numpy() - self.env.mu_star[t]))),
            "mu_mean": float(np.mean(a["mu"].numpy())),
            "std_x": float(self.std().numpy()[0]),
            "std_mu": float(self.std().numpy()[1]),
        }

    def train(self, out_dir: str) -> Dict[str, list]:
        os.makedirs(out_dir, exist_ok=True)
        cfg = self.cfg
        history = {"iteration": [], "mean_return": [], "seconds": [], "env_steps": [], "eval": []}
        start = time.time()
        for iteration in range(cfg["iterations"]):
            # Prompts: states visited by the current deterministic policy
            trajectory = self.simulate(self.env.t_max)
            index = tf.random.uniform([cfg["prompts"]], 0, self.env.t_max, dtype=tf.int32)
            states, actions, log_probs, rewards = self.rollout(tf.gather(trajectory, index))
            advantage = self.advantages(rewards)

            flat = [tf.reshape(x, (-1,) + tuple(x.shape[2:])) for x in (states, actions, log_probs, advantage)]
            n = flat[0].shape[0]
            for _ in range(cfg["update_epochs"]):
                order = tf.random.shuffle(tf.range(n))
                for batch in tf.split(order, cfg["minibatches"]):
                    self._update(*[tf.gather(x, batch) for x in flat])

            history["iteration"].append(iteration)
            history["mean_return"].append(float(tf.reduce_mean(tf.reduce_sum(rewards * self.env.beta ** tf.range(cfg["horizon"], dtype=tf.float32), axis=1))))
            history["seconds"].append(time.time() - start)
            history["env_steps"].append((iteration + 1) * self.samples_per_iteration)

            last = iteration == cfg["iterations"] - 1
            if iteration % cfg["eval_every"] == 0 or last:
                ev = self.quick_eval()
                history["eval"].append({"iteration": iteration, **ev})
                print(
                    f"iter {iteration:5d}  x_err {ev['x_err']:.4f}  mu_err {ev['mu_err']:.4f}  mu {ev['mu_mean']:.2f}  "
                    f"std {ev['std_x']:.3f}/{ev['std_mu']:.3f}  {history['seconds'][-1] / 60:.1f} min",
                    flush=True,
                )
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump(history, f)

        self.network.save_weights(os.path.join(out_dir, "policy.weights.h5"))
        np.save(os.path.join(out_dir, "log_std.npy"), self.log_std.numpy())
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        return history
