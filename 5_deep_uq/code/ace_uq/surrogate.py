"""Uncertainty quantification on top of the pseudo-state DEQN.

Following Deep Uncertainty Quantification, a quantity of interest (here the SCC)
is mapped from the uncertain parameters with a Gaussian-process surrogate. The
design points are chosen by Bayesian active learning: start from a small space-
filling design and repeatedly add the candidate with the largest posterior
standard deviation. The surrogate then gives the distribution of the SCC under
a belief over the parameters, and Sobol indices that attribute its variance.
"""

import warnings
from typing import Callable, Dict

import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# The fitted noise level sits at its lower bound: the DEQN is a deterministic function
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def to_unit(parameters: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (parameters - low) / (high - low)


def from_unit(u: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + (high - low) * u


def sample_belief(belief: Dict[str, Dict], names, n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent draws from the belief distribution of each parameter."""
    columns = []
    for name in names:
        b = belief[name]
        u = rng.random(n)
        if b["distribution"] == "uniform":
            columns.append(b["low"] + (b["high"] - b["low"]) * u)
        elif b["distribution"] == "loguniform":
            columns.append(np.exp(np.log(b["low"]) + (np.log(b["high"]) - np.log(b["low"])) * u))
        else:
            raise ValueError(f"Unknown distribution {b['distribution']}")
    return np.stack(columns, axis=1)


class ActiveGP:
    """GP regression of log f on the unit cube, with design points chosen by active learning."""

    def __init__(self, f: Callable[[np.ndarray], np.ndarray], low: np.ndarray, high: np.ndarray, seed: int = 0):
        self.f = f
        self.low, self.high = low, high
        self.dim = len(low)
        self.rng = np.random.default_rng(seed)
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2))
        kernel += WhiteKernel(1e-6, noise_level_bounds=(1e-10, 1e-2))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed)
        self.U = np.zeros((0, self.dim))
        self.y = np.zeros(0)
        self.trace = []

    def _add(self, U: np.ndarray) -> None:
        values = self.f(from_unit(U, self.low, self.high))
        self.U = np.concatenate([self.U, U])
        self.y = np.concatenate([self.y, np.log(values)])
        self.gp.fit(self.U, self.y)

    def fit(self, n_initial: int, n_total: int, candidates: int = 4096, monitor: Callable = None) -> None:
        self._add(qmc.Sobol(self.dim, seed=self.rng.integers(1 << 31)).random(n_initial))
        pool = qmc.Sobol(self.dim, seed=self.rng.integers(1 << 31)).random(candidates)
        while True:
            if monitor is not None:
                self.trace.append({"points": len(self.y), **monitor(self)})
            if len(self.y) >= n_total:
                break
            _, std = self.gp.predict(pool, return_std=True)
            self._add(pool[[int(np.argmax(std))]])

    def predict(self, parameters: np.ndarray, return_std: bool = False):
        """Surrogate of f (not log f) at parameter vectors in physical units."""
        mean, std = self.gp.predict(to_unit(parameters, self.low, self.high), return_std=True)
        return (np.exp(mean), std) if return_std else np.exp(mean)


def sobol_indices(f: Callable[[np.ndarray], np.ndarray], sample: Callable[[int], np.ndarray], n: int) -> Dict[str, np.ndarray]:
    """First-order (Saltelli 2010) and total (Jansen) Sobol indices of f."""
    A, B = sample(n), sample(n)
    fA, fB = f(A), f(B)
    variance = np.var(np.concatenate([fA, fB]))
    first, total = [], []
    for i in range(A.shape[1]):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fABi = f(ABi)
        first.append(np.mean(fB * (fABi - fA)) / variance)
        total.append(0.5 * np.mean((fA - fABi) ** 2) / variance)
    return {"first": np.array(first), "total": np.array(total)}
