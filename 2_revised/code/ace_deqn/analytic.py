"""Closed-form solution of ACE-DICE, used as the benchmark for the DEQN.

With E_BAU taken as given by the planner, log F is linear in log capital and the
value function is affine in the states. The costates are then constant:

    lambda_k   = beta kappa / (1 - beta kappa)
    lambda_tau = -beta xi_0 (1 + lambda_k) (I - beta sigma^T)^{-1} e_1
    lambda_M   = beta (sigma_forc / M_pre) lambda_tau_1 (I - beta Phi^T)^{-1} e_1

and the controls follow from the first-order conditions:

    x = 1 - beta kappa
    (1 + lambda_k) dlogF/dE = -zeta lambda_M1   (abatement rate capped at 1).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import tensorflow as tf

from ace_deqn.model import AceDice


@dataclass
class AnalyticCostates:
    x: float
    lambda_k: float
    lambda_M: np.ndarray
    lambda_tau: np.ndarray
    carbon_price: float  # -zeta lambda_M1 / (1 + lambda_k), shadow price of emissions in log output


def analytic_costates(model: AceDice) -> AnalyticCostates:
    beta, kappa = model.beta, model.kappa
    lambda_k = beta * kappa / (1.0 - beta * kappa)
    e1_tau = np.array([1.0, 0.0])
    e1_M = np.array([1.0, 0.0, 0.0])
    lambda_tau = -beta * model.xi_0 * (1.0 + lambda_k) * np.linalg.solve(
        np.eye(2) - beta * model.sigma_np.T, e1_tau
    )
    lambda_M = beta * model.sigma_forc / model.M_pre * lambda_tau[0] * np.linalg.solve(
        np.eye(3) - beta * model.Phi_np.T, e1_M
    )
    return AnalyticCostates(
        x=1.0 - beta * kappa,
        lambda_k=lambda_k,
        lambda_M=lambda_M,
        lambda_tau=lambda_tau,
        carbon_price=-model.zeta * lambda_M[0] / (1.0 + lambda_k),
    )


def optimal_abatement(theta_1, E_BAU, theta_2: float, carbon_price: float) -> np.ndarray:
    """Solves dlogF/dE(mu) = carbon_price for the abatement rate mu by bisection.

    dlogF/dE = theta_1 theta_2 mu^(theta_2 - 1) / (E_BAU (1 - theta_1 mu^theta_2))
    is increasing in mu, so the solution is unique. If even mu = 1 leaves the
    marginal product below the carbon price, the constraint E >= 0 binds.
    """
    theta_1 = np.asarray(theta_1, dtype=float)
    E_BAU = np.asarray(E_BAU, dtype=float)

    def marginal_product(mu):
        return theta_1 * theta_2 * mu ** (theta_2 - 1) / (E_BAU * (1 - theta_1 * mu**theta_2))

    lo = np.zeros_like(theta_1)
    hi = np.ones_like(theta_1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_low = marginal_product(mid) < carbon_price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def analytic_actions(model: AceDice, s: tf.Tensor, V: tf.Tensor = None) -> Dict[str, tf.Tensor]:
    """The analytic controls and costates at a batch of states, as decoded actions."""
    c = analytic_costates(model)
    batch = s.shape[0]
    t = model._period(s).numpy()
    E_BAU = model.E_BAU(s).numpy()
    mu = optimal_abatement(model.exo.theta_1[t], E_BAU, model.theta_2, c.carbon_price)
    # Keep emissions strictly inside (0, E_BAU), as the network parameterization does
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    # Where E >= 0 binds, lambda_E absorbs the gap in the emissions FOC
    dlogF_dE = model.dlogF_dE(s, tf.constant(E_BAU * (1 - mu), tf.float32)).numpy()
    gap = (1 + c.lambda_k) * (c.carbon_price - dlogF_dE)
    lambda_E = np.where(mu >= 1 - 1e-6, gap, 0.0)

    def full(value):
        return tf.constant(np.broadcast_to(value, (batch,) + np.shape(value)), tf.float32)

    return {
        "x": full(c.x),
        "E": tf.constant(E_BAU * (1 - mu), tf.float32),
        "V": V if V is not None else tf.zeros(batch),
        "lambda_k": full(c.lambda_k),
        "lambda_M": full(c.lambda_M),
        "lambda_tau": full(c.lambda_tau),
        "lambda_BAU": tf.zeros(batch),
        "lambda_E": tf.constant(lambda_E, tf.float32),
    }


def analytic_trajectory(model: AceDice, n: int = None) -> Dict[str, np.ndarray]:
    """Simulates the analytic optimum from the initial state.

    The value function along the path is the discounted sum of future utility,
    truncated after the exogenous paths end (beta^200 is negligible).
    """
    n = n or model.n_periods - 1
    s = model.initial_state(1)
    states, actions, utility = [], [], []
    for _ in range(n):
        a = analytic_actions(model, s)
        states.append(s.numpy()[0])
        actions.append({k: v.numpy()[0] for k, v in a.items()})
        utility.append(model.log_utility(s, a).numpy()[0])
        s = model.step(s, a)

    utility = np.array(utility)
    V = np.zeros(n)
    running = 0.0
    for t in reversed(range(n)):
        running = utility[t] + model.beta * running
        V[t] = running

    return {"states": np.array(states), "actions": actions, "utility": utility, "V": V}
