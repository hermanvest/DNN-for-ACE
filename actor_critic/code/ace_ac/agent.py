"""Model-based actor-critic for ACE.

The critic V_phi(s) is trained on the Bellman equation of the actor's policy,

    V_phi(s) ~ u(s, pi(s)) + beta V_target(f(s, pi(s))),

and the actor maximizes the one-step lookahead through the known, differentiable
dynamics f,

    max_theta  u(s, pi_theta(s)) + beta V_phi(f(s, pi_theta(s))).

Nothing in the objective is derived by hand: there are no first-order or
envelope conditions and no costate outputs. The costates, and hence the social
cost of carbon, are recovered afterwards as beta times the gradient of the
critic at the next state.
"""

import json
import os
import time
from typing import Dict

import numpy as np
import tensorflow as tf

from ace_ac.env import EPSILON, AceEnv


def mlp(config: Dict, outputs: int, name: str) -> tf.keras.Model:
    net = config["network"]
    initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform", seed=net["seed"]
    )
    inputs = tf.keras.Input(shape=(7,))
    h = inputs
    for _ in range(net["hidden_layers"]):
        h = tf.keras.layers.Dense(net["hidden_nodes"], activation=net["activation"], kernel_initializer=initializer)(h)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(outputs, kernel_initializer=initializer)(h), name=name)


class ActorCritic:
    def __init__(self, env: AceEnv, config: Dict) -> None:
        self.env = env
        self.config = config
        self.train_cfg = config["training"]
        self.value_scale = float(config["network"]["value_scale"])
        self.value_offset = float(config["network"]["value_offset"])
        self.gradient_weight = float(config["training"]["gradient_weight"])
        # Running mean square of the target value gradient, per state component
        self.gradient_ms = tf.Variable(tf.ones(6))

        self.actor = mlp(config, 2, "actor")
        self.critic = mlp(config, 1, "critic")
        self.target = mlp(config, 1, "target_critic")
        self.target.set_weights(self.critic.get_weights())

        exploration = config["exploration"]
        self.noise = {k: float(v) for k, v in exploration.items() if k.startswith("sd_")}
        self.copies = int(exploration["copies"])

        states_per_iteration = env.t_max * (1 + self.copies)
        steps = self.train_cfg["iterations"] * self.train_cfg["epochs"] * int(
            np.ceil(states_per_iteration / self.train_cfg["batch_size"])
        )

        def optimizer(rate):
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                rate, decay_steps=steps, alpha=self.train_cfg["final_learning_rate_factor"]
            )
            return tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=self.train_cfg["clipnorm"])

        self.actor_optimizer = optimizer(self.train_cfg["actor_learning_rate"])
        self.critic_optimizer = optimizer(self.train_cfg["critic_learning_rate"])

    ################ NETWORKS ################
    def policy(self, s: tf.Tensor) -> Dict[str, tf.Tensor]:
        raw = self.actor(self.env.preprocess(s))
        squash = lambda z: EPSILON + (1 - 2 * EPSILON) * tf.sigmoid(z)
        return {"x": squash(raw[:, 0]), "mu": squash(raw[:, 1])}

    def value(self, s: tf.Tensor, network: tf.keras.Model = None) -> tf.Tensor:
        network = network or self.critic
        return self.value_offset + self.value_scale * network(self.env.preprocess(s))[:, 0]

    def costates(self, s_next: tf.Tensor) -> tf.Tensor:
        """beta times the gradient of the critic w.r.t. the next-period state (k, M, tau)."""
        with tf.GradientTape() as tape:
            tape.watch(s_next)
            v = self.value(s_next)
        return self.env.beta * tape.gradient(v, s_next)[:, :6]

    ################ TRAINING ################
    @tf.function
    def _simulate_step(self, s: tf.Tensor) -> tf.Tensor:
        return self.env.step(s, self.policy(s))

    def simulate(self, n: int) -> tf.Tensor:
        s = self.env.initial_state(1)
        states = []
        for _ in range(n):
            states.append(s)
            s = self._simulate_step(s)
        return tf.concat(states, axis=0)

    def explore(self, states: tf.Tensor) -> tf.Tensor:
        """Trajectory states plus perturbed copies; time is never perturbed."""
        batches = [states]
        for _ in range(self.copies):
            shape = (states.shape[0], 1)
            k = states[:, 0:1] + self.noise["sd_k"] * tf.random.normal(shape)
            M = states[:, 1:4] * (1 + self.noise["sd_M"] * tf.random.normal((states.shape[0], 3)))
            tau = states[:, 4:6] * (1 + self.noise["sd_tau"] * tf.random.normal((states.shape[0], 2)))
            M = tf.maximum(M, 1.0)
            tau = tf.maximum(tau, 1.0)  # Temperatures at or above preindustrial
            batches.append(tf.concat([k, M, tau, states[:, 6:7]], axis=1))
        return tf.concat(batches, axis=0)

    @tf.function
    def _critic_step(self, s: tf.Tensor):
        beta = self.env.beta

        # Critic: Bellman equation of the current policy, with a slowly moving target.
        # The same equation, differentiated w.r.t. the state by autodiff, gives a target
        # for the gradient of the value function (value-gradient learning).
        with tf.GradientTape() as state_tape:
            state_tape.watch(s)
            a = self.policy(s)
            target = self.env.utility(s, a) + beta * self.value(self.env.step(s, a), self.target)
        target_gradient = tf.stop_gradient(state_tape.gradient(target, s)[:, :6])
        target = tf.stop_gradient(target)

        # Normalize each gradient component by the running RMS of its target, so that
        # small but decisive gradients (such as dV/dM) carry as much weight as large ones
        self.gradient_ms.assign(0.99 * self.gradient_ms + 0.01 * tf.reduce_mean(tf.square(target_gradient), axis=0))
        gradient_scale = tf.sqrt(self.gradient_ms) + 1e-12

        with tf.GradientTape() as tape:
            with tf.GradientTape() as inner:
                inner.watch(s)
                v = self.value(s)
            gradient = inner.gradient(v, s)[:, :6]
            td_loss = tf.reduce_mean(tf.square(v - target))
            gradient_loss = tf.reduce_mean(tf.reduce_sum(tf.square((gradient - target_gradient) / gradient_scale), axis=1))
            critic_loss = td_loss + self.gradient_weight * gradient_loss
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        rate = self.train_cfg["target_update"]
        for w_target, w in zip(self.target.trainable_variables, self.critic.trainable_variables):
            w_target.assign((1 - rate) * w_target + rate * w)
        return td_loss

    @tf.function
    def _actor_step(self, s: tf.Tensor):
        beta = self.env.beta
        # Actor: maximize reward plus discounted value, differentiating through the dynamics
        with tf.GradientTape() as tape:
            a = self.policy(s)
            objective = tf.reduce_mean(self.env.utility(s, a) + beta * self.value(self.env.step(s, a)))
            actor_loss = -objective
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return objective

    def quick_eval(self) -> Dict[str, float]:
        """Errors of the current policy against the analytic one along its own trajectory."""
        s = self.simulate(self.env.t_max)
        a = self.policy(s)
        horizon = 58  # periods up to about 2300
        t = s[:horizon, 6].numpy().astype(int)
        lam = self.costates(self.env.step(s[:horizon], {k: v[:horizon] for k, v in a.items()})).numpy()
        c = self.env.costates
        return {
            "x_err": float(np.max(np.abs(a["x"][:horizon].numpy() - c.x))),
            "mu_err": float(np.mean(np.abs(a["mu"][:horizon].numpy() - self.env.mu_star[t]))),
            "lambda_k_err": float(np.mean(np.abs(lam[:, 0] / c.lambda_k - 1))),
            "lambda_M1_err": float(np.mean(np.abs(lam[:, 1] / c.lambda_M[0] - 1))),
            "lambda_M1_ratio": float(np.median(lam[:, 1] / c.lambda_M[0])),
            "mu_mean": float(np.mean(a["mu"][:horizon].numpy())),
        }

    def train(self, out_dir: str) -> Dict[str, list]:
        os.makedirs(out_dir, exist_ok=True)
        cfg = self.train_cfg
        history = {"iteration": [], "critic_loss": [], "objective": [], "seconds": [], "eval": []}
        start = time.time()
        for iteration in range(cfg["iterations"]):
            states = self.explore(self.simulate(self.env.t_max))
            losses, objectives = [], []
            update_actor = iteration >= cfg["critic_warmup"]
            step = 0
            for _ in range(cfg["epochs"]):
                for batch in tf.data.Dataset.from_tensor_slices(tf.random.shuffle(states)).batch(cfg["batch_size"]):
                    losses.append(float(self._critic_step(batch)))
                    # The actor moves less often than the critic, so it follows a settled critic
                    if update_actor and step % cfg["actor_delay"] == 0:
                        objectives.append(float(self._actor_step(batch)))
                    step += 1
            history["iteration"].append(iteration)
            history["critic_loss"].append(float(np.mean(losses)))
            history["objective"].append(float(np.mean(objectives)) if objectives else None)
            history["seconds"].append(time.time() - start)

            last = iteration == cfg["iterations"] - 1
            if iteration % cfg["eval_every"] == 0 or last:
                ev = self.quick_eval()
                history["eval"].append({"iteration": iteration, **ev})
                print(
                    f"iter {iteration:5d}  critic {history['critic_loss'][-1]:.2e}  "
                    f"x_err {ev['x_err']:.4f}  mu_err {ev['mu_err']:.4f}  "
                    f"lk_err {ev['lambda_k_err']:.3f}  lM1_ratio {ev['lambda_M1_ratio']:+.2f}  mu {ev['mu_mean']:.2f}  "
                    f"{history['seconds'][-1] / 60:.1f} min",
                    flush=True,
                )
                with open(os.path.join(out_dir, "history.json"), "w") as f:
                    json.dump(history, f)

        self.actor.save_weights(os.path.join(out_dir, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(out_dir, "critic.weights.h5"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f)
        return history
