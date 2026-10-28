"""The closed-form benchmark is consistent with the environment's dynamics."""

import os

import numpy as np
import pytest
import tensorflow as tf
import yaml

from ace_ac.env import AceEnv

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture(params=["2016", "2023"], scope="module")
def env(request):
    with open(os.path.join(CONFIG_DIR, f"ace_dice_{request.param}.yaml")) as f:
        return AceEnv(yaml.safe_load(f))


def test_reference_path_matches_dice_definition(env):
    """Along the analytic path, the exogenous E_BAU equals sigma_t * Y_gross(K_t)."""
    s = env.initial_state(1)
    for t in range(60):
        E_dice = env.exo.sigma[t] * float(tf.exp(env.log_Y_gross(s))[0])
        assert float(env.E_BAU(s)[0]) == pytest.approx(E_dice, rel=1e-4)
        s = env.step(s, env.analytic_action(s))


def test_costates_are_discounted_value_gradients(env):
    """beta * dV*/ds at the next state equals the analytic costates (finite differences)."""
    s0 = env.initial_state(1)
    s1 = env.step(s0, env.analytic_action(s0)).numpy().astype(np.float64)
    c = env.costates
    expected = {0: c.lambda_k, 1: c.lambda_M[0], 2: c.lambda_M[1], 4: c.lambda_tau[0]}
    steps = {0: 0.05, 1: 20.0, 2: 20.0, 4: 0.02}
    for i, lam in expected.items():
        up, down = s1.copy(), s1.copy()
        up[0, i] += steps[i]
        down[0, i] -= steps[i]
        values = env.rollout_value(tf.constant(np.vstack([up, down]), tf.float32), env.analytic_action, n=250).numpy()
        gradient = (values[0] - values[1]) / (2 * steps[i])
        assert env.beta * gradient == pytest.approx(lam, rel=0.03), i


def test_analytic_policy_beats_perturbations(env):
    """Small changes to the analytic consumption or abatement rate lower welfare."""
    s0 = env.initial_state(1)
    best = float(env.rollout_value(s0, env.analytic_action)[0])

    def perturbed(dx, dmu):
        def policy(s):
            a = env.analytic_action(s)
            return {"x": a["x"] + dx, "mu": tf.clip_by_value(a["mu"] + dmu, 1e-6, 1 - 1e-6)}
        return policy

    for dx, dmu in [(0.02, 0), (-0.02, 0), (0, 0.05), (0, -0.05)]:
        assert float(env.rollout_value(s0, perturbed(dx, dmu))[0]) < best
