"""The closed form must satisfy every equilibrium condition, for any parameter draw."""

import os

import numpy as np
import pytest
import tensorflow as tf
import yaml

from ace_uq.analytic import analytic_actions, analytic_costates, analytic_trajectories
from ace_uq.model import AceDiceUQ
from ace_uq.residuals import RESIDUAL_NAMES, residuals

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def load_model(version: str) -> AceDiceUQ:
    with open(os.path.join(CONFIG_DIR, f"ace_dice_{version}.yaml")) as f:
        return AceDiceUQ(yaml.safe_load(f))


@pytest.fixture(params=["2016", "2023"], scope="module")
def model(request):
    return load_model(request.param)


def draws(model: AceDiceUQ, n: int = 8) -> np.ndarray:
    corners = np.array([model.low, model.high, model.baseline])
    return np.concatenate([corners, model.sample_parameters(n, np.random.default_rng(0))])


def test_carbon_matrix_conserves_mass(model):
    for u in (model.low[2], 1.0, model.high[2]):
        np.testing.assert_allclose(model.Phi_np(u).sum(axis=0), 1.0, atol=1e-12)


def test_tensor_matrix_matches_numpy(model):
    p = draws(model)
    Phi = model.Phi(model.initial_state(p)).numpy()
    for i, u in enumerate(p[:, 2]):
        np.testing.assert_allclose(Phi[i], model.Phi_np(u), atol=1e-6)


def test_baseline_matches_part_2(model):
    """At the baseline parameters the costates equal those of the single-instance model."""
    c = analytic_costates(model, model.baseline)
    beta = (1 + model.baseline[0]) ** -model.timestep
    assert c["x"][0] == pytest.approx(1 - beta * model.kappa)
    if model.version == "2016":
        assert c["x"][0] == pytest.approx(0.7395, abs=1e-4)


def test_analytic_paths_have_zero_residuals(model):
    p = draws(model)
    path = analytic_trajectories(model, p, n=model.t_max + 1)
    n = model.t_max
    s = tf.constant(path["states"][:, :n].reshape(-1, 10), tf.float32)
    s_next = tf.constant(path["states"][:, 1 : n + 1].reshape(-1, 10), tf.float32)
    V = path["V"]
    a = analytic_actions(model, s, tf.constant(V[:, :n].reshape(-1), tf.float32))
    a_next = analytic_actions(model, s_next, tf.constant(V[:, 1 : n + 1].reshape(-1), tf.float32))

    np.testing.assert_allclose(model.step(s, a).numpy(), s_next.numpy(), rtol=1e-4)

    r = np.abs(residuals(model, s, a, s_next, a_next).numpy())
    worst = dict(zip(RESIDUAL_NAMES, r.max(axis=0)))
    # The Bellman residual is the largest: V reaches a few hundred when prtp is low
    assert max(worst.values()) < 2e-3, worst


def test_scc_rises_as_discount_rate_falls(model):
    p = np.array([[0.005, model.baseline[1], 1.0], [0.035, model.baseline[1], 1.0]])
    price = analytic_costates(model, p)["carbon_price"]
    assert price[0] > 3 * price[1]
