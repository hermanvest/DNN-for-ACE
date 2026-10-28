"""Corrected equilibrium conditions (thesis eqs. 5.13-5.20 and A.4-A.11).

Each residual is zero at the optimum. Residuals involving small costates are
divided by the matching output scale so that all conditions carry comparable
weight in the policy objective; this rescaling does not move the optimum.
"""

from typing import Dict

import tensorflow as tf

from ace_deqn.model import AceDice

RESIDUAL_NAMES = [
    "l0 capital",
    "l1 M1",
    "l1 M2",
    "l1 M3",
    "l2 tau1",
    "l2 tau2",
    "l3 consumption",
    "l4 emissions",
    "l5A E>=0",
    "l5B E<=E_BAU",
    "l6 Bellman",
]


def fischer_burmeister(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return a + b - tf.sqrt(tf.square(a) + tf.square(b))


def residuals(
    model: AceDice,
    s: tf.Tensor,
    a: Dict[str, tf.Tensor],
    s_next: tf.Tensor,
    a_next: Dict[str, tf.Tensor],
) -> tf.Tensor:
    """Returns the residuals for a batch of transitions, shape [batch, 11]."""
    beta = model.beta
    scale = model.scale

    lambda_k, lambda_k_next = a["lambda_k"], a_next["lambda_k"]
    lambda_M, lambda_M_next = a["lambda_M"], a_next["lambda_M"]
    lambda_tau, lambda_tau_next = a["lambda_tau"], a_next["lambda_tau"]
    x, E = a["x"], a["E"]

    # l0: capital costate, lambda_k = beta (1 + lambda_k') dlogF'/dk'
    l0 = lambda_k - beta * (1.0 + lambda_k_next) * model.dlogF_dk()

    # l1: carbon costates, lambda_M = beta (Phi^T lambda_M' + e1 lambda_tau1' sigma_forc / M_pre)
    forcing_M = lambda_tau_next[:, 0:1] * model.sigma_forc / model.M_pre
    l1 = lambda_M - beta * (
        tf.matmul(lambda_M_next, model.Phi)
        + forcing_M * tf.constant([[1.0, 0.0, 0.0]])
    )
    l1 /= scale["lambda_M"]

    # l2: temperature costates, lambda_tau = beta (sigma^T lambda_tau' - e1 xi_0 (1 + lambda_k'))
    damages = model.xi_0 * (1.0 + lambda_k_next[:, None])
    l2 = lambda_tau - beta * (
        tf.matmul(lambda_tau_next, model.sigma)
        - damages * tf.constant([[1.0, 0.0]])
    )
    l2 /= scale["lambda_tau"]

    # l3: consumption rate, 1/x = lambda_k / (1 - x)
    l3 = 1.0 / x - lambda_k / (1.0 - x)

    # l4: emissions, (1 + lambda_k) dlogF/dE + zeta lambda_M1 + lambda_E - lambda_BAU = 0
    l4 = (
        (1.0 + lambda_k) * model.dlogF_dE(s, E)
        + model.zeta * lambda_M[:, 0]
        + a["lambda_E"]
        - a["lambda_BAU"]
    )
    l4 /= scale["lambda_M"]

    # l5: KKT conditions on 0 <= E <= E_BAU, written in scale-free units
    E_share = E / model.E_BAU(s)
    l5A = fischer_burmeister(a["lambda_E"] / scale["lambda_kkt"], E_share)
    l5B = fischer_burmeister(a["lambda_BAU"] / scale["lambda_kkt"], 1.0 - E_share)

    # l6: Bellman equation
    l6 = model.log_utility(s, a) + beta * a_next["V"] - a["V"]

    return tf.concat(
        [
            l0[:, None],
            l1,
            l2,
            l3[:, None],
            l4[:, None],
            l5A[:, None],
            l5B[:, None],
            l6[:, None],
        ],
        axis=1,
    )
