"""SDDP and its stage problems must reproduce the closed form."""

import numpy as np
import pytest

from ace_sddp.experiments import load_config
from ace_sddp.model import Economy, Instance, optimal_abatement
from ace_sddp.neural import decode, encode
from ace_sddp.sddp import SDDP
from ace_sddp.stage import StageProblem


@pytest.fixture(params=["2016", "2023"], scope="module")
def setup(request):
    cfg = load_config(request.param)
    eco = Economy(cfg)
    return cfg, eco, StageProblem(cfg["sddp"]["max_cuts"], cfg["sddp"]["consumption_bounds"])


def test_closed_form_cut_is_a_fixed_point(setup):
    """With the closed-form cut for V_t+1, a stage returns the closed-form policy and the same slopes."""
    cfg, eco, stage = setup
    inst = Instance(eco, [0.01, 0.03, 0.9])
    cf = inst.closed_form()
    cut = np.concatenate([[0.0], cf.slopes])[None, :]
    for t in (0, 10, 40):
        s = inst.closed_form_path(t + 1)["states"][t]
        sol = stage.solve(inst, t, s, cut)
        assert sol.x == pytest.approx(cf.x, abs=1e-4)
        assert sol.mu == pytest.approx(optimal_abatement(eco.theta_1[t], eco.E_BAU[t], eco.theta_2, cf.carbon_price), abs=1e-3)
        np.testing.assert_allclose(sol.gradient, cf.slopes, rtol=1e-5)


def test_sddp_matches_closed_form_at_baseline(setup):
    cfg, eco, stage = setup
    inst = Instance(eco, eco.baseline)
    result = SDDP(inst, stage, cfg).run()
    closed = inst.closed_form_path()
    n = eco.evaluate
    assert result.iterations == 2
    assert result.upper - result.lower < 1e-5 * abs(result.lower)
    assert np.max(np.abs(result.path["scc"][:n] / closed["scc"][:n] - 1)) < 1e-4
    assert result.lower == pytest.approx(closed["value"], rel=1e-7)


def test_cut_encoding_round_trips():
    cuts = np.array([[12.0, 0.4, -1e-4, -2e-5, -5e-8, -0.06, -0.05], [-3.0, 0.3, 0.0, -1e-6, -1e-9, -0.1, -0.02]])
    np.testing.assert_allclose(decode(encode(cuts)), cuts, atol=1e-11)
