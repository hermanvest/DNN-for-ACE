import pytest
import numpy as np

from ..equations_of_motion import DICE_Prod_Stateupdater

config_dict = {
    "production_specific": {
        "xi_0": 1.0,
        "kappa": 1.0,
        "delta_k": 1.0,
        "g_k": 1.0,
        "g_0_sigma": 1.0,
        "delta_sigma": 1.0,
        "sigma_0": 1.0,
        "delta_t": 1.0,
        "p_0_back": 1.0,
        "g_back": 1.0,
        "theta_2": 1.0,
        "c2co2": 1.0,
    },
    "climate_specific": {
        "sigma_forc": 1.0,
        "M_pre": 100.0,
        "M": [100.0, 100.0, 100.0],
        "Phi": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "sigma": [
            [1, 1],
            [1, 1],
        ],
    },
}


@pytest.fixture
def setUp():
    return DICE_Prod_Stateupdater(config_dict)


###################### Type tests ######################


def test_k_t_plus_is_float(setUp):
    assert isinstance(setUp.k_tplus(1.0, 1.0, 0.1), float)


def test_m_1_plus_is_float(setUp):
    m2 = np.array([1.0, 1.0, 1.0])
    assert isinstance(setUp.m_1plus(m2, 1.0), float)


def test_m_2_plus_is_float(setUp):
    m = np.array([1.0, 1.0, 1.0])
    assert isinstance(setUp.m_2plus(m), float)


def test_m_3_plus_is_float(setUp):
    m = np.array([1.0, 1.0, 1.0])
    assert isinstance(setUp.m_3plus(m), float)


def test_tau_1_plus_is_float(setUp):
    tau = np.array([1.0, 1.0])
    assert isinstance(setUp.tau_1plus(tau, 1.0), float)


def test_tau_2_plus_is_float(setUp):
    tau = np.array([1.0, 1.0])
    assert isinstance(setUp.tau_2plus(tau), float)


###################### Calculation tests ######################


def test_k_tplus_calculations(setUp):
    expecting = 0.8946
    assert np.isclose(setUp.k_tplus(1.0, 1.0, 0.1), expecting, rtol=1e-3)


def test_m_1_plus_calculations(setUp):
    expecting = 4.0
    m = np.array([1.0, 1.0, 1.0])
    assert np.isclose(setUp.m_1plus(m, 1), expecting, rtol=1e-3)


def test_m_2_plus_calculations(setUp):
    expecting = 3.0
    m = np.array([1.0, 1.0, 1.0])
    assert np.isclose(setUp.m_2plus(m), expecting, rtol=1e-3)


def test_m_3_plus_calculations(setUp):
    expecting = 3.0
    m = np.array([1.0, 1.0, 1.0])
    assert np.isclose(setUp.m_3plus(m), expecting, rtol=1e-3)


def test_tau_1_plus_calculations(setUp):
    expecting = 2.01
    tau = np.array([1.0, 1.0])
    assert np.isclose(setUp.tau_1plus(tau, 1.0), expecting, rtol=1e-3)


def test_tau_2_plus_calculations(setUp):
    expecting = 2.0
    tau = np.array([1.0, 1.0])
    assert np.isclose(setUp.tau_2plus(tau), expecting, rtol=1e-3)
