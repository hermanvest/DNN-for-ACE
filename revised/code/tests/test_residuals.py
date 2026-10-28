"""The analytic solution must satisfy every corrected equilibrium condition."""

import os

import numpy as np
import pytest
import tensorflow as tf
import yaml

from ace_deqn.analytic import analytic_actions, analytic_costates, analytic_trajectory
from ace_deqn.model import AceDice
from ace_deqn.residuals import RESIDUAL_NAMES, residuals

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def load_model(version: str) -> AceDice:
    with open(os.path.join(CONFIG_DIR, f"ace_dice_{version}.yaml")) as f:
        return AceDice(yaml.safe_load(f))


@pytest.fixture(params=["2016", "2023"], scope="module")
def model(request):
    return load_model(request.param)


def test_consumption_rate_matches_closed_form(model):
    c = analytic_costates(model)
    assert c.x == pytest.approx(1 - model.beta * model.kappa)
    assert c.lambda_k == pytest.approx((1 - c.x) / c.x)


def test_analytic_path_has_zero_residuals(model):
    path = analytic_trajectory(model)
    n = model.t_max
    s = tf.constant(path["states"][:n], tf.float32)
    s_next = tf.constant(path["states"][1 : n + 1], tf.float32)
    a = analytic_actions(model, s, tf.constant(path["V"][:n], tf.float32))
    a_next = analytic_actions(model, s_next, tf.constant(path["V"][1 : n + 1], tf.float32))

    # The analytic actions must reproduce the simulated transition
    np.testing.assert_allclose(model.step(s, a).numpy(), s_next.numpy(), rtol=1e-4)

    r = np.abs(residuals(model, s, a, s_next, a_next).numpy())
    worst = dict(zip(RESIDUAL_NAMES, r.max(axis=0)))
    # Float32 arithmetic on values of order 1-100 leaves residuals around 1e-4
    assert max(worst.values()) < 1e-3, worst


def test_original_conditions_fail_at_optimum(model):
    """The capital and consumption conditions of the original code are not zero at the optimum."""
    c = analytic_costates(model)
    beta, kappa = model.beta, model.kappa
    original_l0 = beta * c.lambda_k - c.lambda_k
    original_l3 = 1 / c.x - (c.lambda_k + beta * c.lambda_k * kappa) / (1 - c.x)
    assert abs(original_l0) > 1e-2
    assert abs(original_l3) > 1e-1
