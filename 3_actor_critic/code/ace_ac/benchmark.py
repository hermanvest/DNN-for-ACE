"""Closed-form solution of ACE-DICE with business-as-usual emissions taken as given.

The costates are constant over time:

    lambda_k   = beta kappa / (1 - beta kappa)
    lambda_tau = -beta xi_0 (1 + lambda_k) (I - beta sigma^T)^{-1} e_1
    lambda_M   = beta (sigma_forc / M_pre) lambda_tau_1 (I - beta Phi^T)^{-1} e_1

and the optimal controls are x = 1 - beta kappa and the abatement rate that sets
the marginal product of emissions equal to the carbon price
-zeta lambda_M1 / (1 + lambda_k), capped at full abatement.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Costates:
    x: float
    lambda_k: float
    lambda_M: np.ndarray
    lambda_tau: np.ndarray
    carbon_price: float  # -zeta lambda_M1 / (1 + lambda_k), in units of log output per GtCO2/yr


def analytic_costates(beta, kappa, xi_0, sigma, Phi, sigma_forc, M_pre, zeta) -> Costates:
    lambda_k = beta * kappa / (1.0 - beta * kappa)
    lambda_tau = -beta * xi_0 * (1.0 + lambda_k) * np.linalg.solve(
        np.eye(2) - beta * sigma.T, np.array([1.0, 0.0])
    )
    lambda_M = beta * sigma_forc / M_pre * lambda_tau[0] * np.linalg.solve(
        np.eye(3) - beta * Phi.T, np.array([1.0, 0.0, 0.0])
    )
    return Costates(
        x=1.0 - beta * kappa,
        lambda_k=lambda_k,
        lambda_M=lambda_M,
        lambda_tau=lambda_tau,
        carbon_price=-zeta * lambda_M[0] / (1.0 + lambda_k),
    )


def optimal_abatement(theta_1, E_BAU, theta_2: float, carbon_price: float) -> np.ndarray:
    """Abatement rate mu solving dlogF/dE = carbon_price, by bisection on [0, 1].

    dlogF/dE = theta_1 theta_2 mu^(theta_2 - 1) / (E_BAU (1 - theta_1 mu^theta_2))
    increases in mu, so the solution is unique; it is 1 where E >= 0 binds.
    """
    theta_1 = np.asarray(theta_1, dtype=float)
    E_BAU = np.asarray(E_BAU, dtype=float)

    def marginal_product(mu):
        return theta_1 * theta_2 * mu ** (theta_2 - 1) / (E_BAU * (1 - theta_1 * mu**theta_2))

    lo, hi = np.zeros_like(theta_1), np.ones_like(theta_1)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        too_low = marginal_product(mid) < carbon_price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)
