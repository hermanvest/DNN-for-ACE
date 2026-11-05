"""The cut network of neural SDDP (Dai et al., 2021), for ACE.

The network maps a problem instance, here the uncertain parameters (prtp, xi_0,
carbon_exchange), and a stage t to the cut that approximates V_t:

    (p, t)  ->  [alpha_t, b_t]

It is trained on cuts from SDDP solutions of training instances. For a new instance, its
cuts either replace SDDP altogether (one forward pass with the predicted cuts), or warm-start
it. Dai et al. predict several cuts in a learned low-dimensional space; in ACE the value
function is affine, so one cut per stage in the original six-dimensional state suffices.

The slopes have fixed signs (capital is valuable, carbon and temperature are harmful) and
span orders of magnitude, so the network predicts their logarithms: the error is then
relative, the same for a low and a high social cost of carbon.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

SIGNS = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])  # Signs of [b_k, b_M1..3, b_tau1..2]
FLOOR = 1e-12  # Slopes at the very last stages are exactly zero


def encode(cuts: np.ndarray) -> np.ndarray:
    """[alpha, b] -> [alpha, log |b|]."""
    return np.column_stack([cuts[:, 0], np.log(np.maximum(SIGNS * cuts[:, 1:], 0.0) + FLOOR)])


def decode(targets: np.ndarray) -> np.ndarray:
    return np.column_stack([targets[:, 0], SIGNS * (np.exp(targets[:, 1:]) - FLOOR)])


class CutNetwork:
    def __init__(self, config: dict, low: np.ndarray, high: np.ndarray, horizon: int):
        self.cfg = config["neural"]
        self.low, self.high, self.horizon = np.asarray(low), np.asarray(high), horizon
        init = tf.keras.initializers.GlorotUniform(seed=self.cfg["seed"])
        inputs = tf.keras.Input(shape=(4,))
        h = inputs
        for _ in range(self.cfg["hidden_layers"]):
            h = tf.keras.layers.Dense(self.cfg["hidden_nodes"], activation=self.cfg["activation"], kernel_initializer=init)(h)
        self.model = tf.keras.Model(inputs, tf.keras.layers.Dense(7, kernel_initializer=init)(h))
        self.mean = np.zeros(7)
        self.std = np.ones(7)

    def features(self, parameters: np.ndarray, t: np.ndarray) -> np.ndarray:
        p_hat = 2.0 * (np.atleast_2d(parameters) - self.low) / (self.high - self.low) - 1.0
        return np.column_stack([p_hat, 2.0 * np.asarray(t) / self.horizon - 1.0]).astype(np.float32)

    def fit(self, parameters: np.ndarray, t: np.ndarray, cuts: np.ndarray) -> dict:
        """Supervised training on SDDP cuts; rows are (parameter vector, stage, cut)."""
        tf.random.set_seed(self.cfg["seed"])
        targets = encode(cuts)
        self.mean, self.std = targets.mean(axis=0), targets.std(axis=0) + 1e-12
        y = ((targets - self.mean) / self.std).astype(np.float32)
        x = self.features(parameters, t)
        steps = self.cfg["epochs"] * int(np.ceil(len(x) / self.cfg["batch_size"]))
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            self.cfg["learning_rate"], steps, alpha=self.cfg["final_learning_rate"] / self.cfg["learning_rate"]
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(schedule), loss="mse")
        history = self.model.fit(x, y, epochs=self.cfg["epochs"], batch_size=self.cfg["batch_size"], verbose=0, shuffle=True)
        return {"loss": [float(v) for v in history.history["loss"]]}

    def predict(self, parameters: np.ndarray, stages: np.ndarray) -> np.ndarray:
        """Cuts [alpha, b] for one parameter vector at the given stages."""
        x = self.features(np.repeat(np.atleast_2d(parameters), len(stages), axis=0), stages)
        y = self.model(x, training=False).numpy() * self.std + self.mean
        return decode(y)

    def cuts_for(self, parameters: np.ndarray) -> list:
        """One predicted cut per stage t = 0..horizon-1, in the format of `SDDP.warm_start`."""
        stages = np.arange(self.horizon)
        return [row[None, :] for row in self.predict(parameters, stages)]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(folder / "cuts.weights.h5")
        (folder / "normalisation.json").write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}))

    def load(self, folder: Path) -> None:
        self.model.load_weights(folder / "cuts.weights.h5")
        norm = json.loads((folder / "normalisation.json").read_text())
        self.mean, self.std = np.array(norm["mean"]), np.array(norm["std"])
