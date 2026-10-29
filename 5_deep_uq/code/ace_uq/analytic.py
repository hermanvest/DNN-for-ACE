"""Closed-form solution of ACE-DICE for any vector of the uncertain parameters.

For given (prtp, xi_0, carbon_exchange) the costates are constant (see part 2):

    lambda_k   = beta kappa / (1 - beta kappa)
    lambda_tau = -beta xi_0 (1 + lambda_k) (I - beta sigma^T)^{-1} e_1
    lambda_M   = beta (sigma_forc / M_pre) lambda_tau_1 (I - beta Phi^T)^{-1} e_1

so the SCC is known in closed form across the whole parameter space. This is
the benchmark for the pseudo-state DEQN and for the surrogate built on it.
"""

from typing import Dict

import numpy as np
import tensorflow as tf

from ace_uq.model import AceDiceUQ


def analytic_costates(model: AceDiceUQ, parameters: np.ndarray) -> Dict[str, np.ndarray]:
    """Costates for a batch of parameter vectors [n, 3]; every entry has leading dimension n."""
    parameters = np.atleast_2d(parameters)
    prtp, xi_0, exchange = parameters.T
    beta = (1.0 + prtp) ** -model.timestep
    lambda_k = beta * model.kappa / (1.0 - beta * model.kappa)
    lambda_tau = np.stack(
        [
            -b * xi * (1.0 + lk) * np.linalg.solve(np.eye(2) - b * model.sigma_np.T, [1.0, 0.0])
            for b, xi, lk in zip(beta, xi_0, lambda_k)
        ]
    )
    lambda_M = np.stack(
        [
            b * model.sigma_forc / model.M_pre * lt
            * np.linalg.solve(np.eye(3) - b * model.Phi_np(u).T, [1.0, 0.0, 0.0])
            for b, lt, u in zip(beta, lambda_tau[:, 0], exchange)
        ]
    )
    return {
        "beta": beta,
        "x": 1.0 - beta * model.kappa,
        "lambda_k": lambda_k,
        "lambda_M": lambda_M,
        "lambda_tau": lambda_tau,
        "carbon_price": -model.zeta * lambda_M[:, 0] / (1.0 + lambda_k),
    }


def optimal_abatement(theta_1, E_BAU, theta_2: float, carbon_price) -> np.ndarray:
    """Solves dlogF/dE(mu) = carbon_price for mu by bisection (capped at mu = 1)."""
    theta_1 = np.asarray(theta_1, dtype=float)
    E_BAU = np.asarray(E_BAU, dtype=float)

    def marginal_product(mu):
        return theta_1 * theta_2 * mu ** (theta_2 - 1) / (E_BAU * (1 - theta_1 * mu**theta_2))

    lo = np.zeros_like(E_BAU)
    hi = np.ones_like(E_BAU)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_low = marginal_product(mid) < carbon_price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def analytic_actions(model: AceDiceUQ, s: tf.Tensor, V: tf.Tensor = None) -> Dict[str, tf.Tensor]:
    """The analytic controls and costates at a batch of states, as decoded actions."""
    c = analytic_costates(model, s[:, 7:].numpy().astype(float))
    batch = s.shape[0]
    t = model._period(s).numpy()
    E_BAU = model.E_BAU(s).numpy()
    mu = optimal_abatement(model.exo.theta_1[t], E_BAU, model.theta_2, c["carbon_price"])
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    dlogF_dE = model.dlogF_dE(s, tf.constant(E_BAU * (1 - mu), tf.float32)).numpy()
    gap = (1 + c["lambda_k"]) * (c["carbon_price"] - dlogF_dE)
    lambda_E = np.where(mu >= 1 - 1e-6, gap, 0.0)

    def const(value):
        return tf.constant(value, tf.float32)

    return {
        "x": const(c["x"]),
        "E": const(E_BAU * (1 - mu)),
        "V": V if V is not None else tf.zeros(batch),
        "lambda_k": const(c["lambda_k"]),
        "lambda_M": const(c["lambda_M"]),
        "lambda_tau": const(c["lambda_tau"]),
        "lambda_BAU": tf.zeros(batch),
        "lambda_E": const(lambda_E),
    }


def analytic_trajectories(model: AceDiceUQ, parameters: np.ndarray, n: int = None) -> Dict[str, np.ndarray]:
    """Simulates the analytic optimum from the initial state for every parameter vector.

    Returns arrays of shape [draws, n, ...]. The value function is the discounted
    sum of future utility, truncated where the exogenous paths end.
    """
    n = n or model.n_periods - 1
    s = model.initial_state(parameters)
    states, actions, utility = [], [], []
    for _ in range(n):
        a = analytic_actions(model, s)
        states.append(s.numpy())
        actions.append({k: v.numpy() for k, v in a.items()})
        utility.append(model.log_utility(s, a).numpy())
        s = model.step(s, a)

    beta = analytic_costates(model, parameters)["beta"]
    utility = np.stack(utility, axis=1)
    V = np.zeros_like(utility)
    running = np.zeros(len(beta))
    for t in reversed(range(n)):
        running = utility[:, t] + beta * running
        V[:, t] = running

    stacked = {k: np.stack([a[k] for a in actions], axis=1) for k in actions[0]}
    return {"states": np.stack(states, axis=1), "actions": stacked, "utility": utility, "V": V}
