import pytest
import numpy as np

from ..state_updates import DICE_Prod_Stateupdater

config_dict = {
    "production_specific": {
        "xi_0": 0.2219,
        "kappa": 0.3,
        "delta_k": 0.44,
        "g_k": 0.29,
        "g_0_sigma": -0.0152,
        "delta_sigma": 0.001,
        "sigma_0": 0.00009556,
        "delta_t": 5,
        "p_0_back": 0.55,
        "g_back": 0.005,
        "theta_2": 2.6,
        "c2co2": 3.666,
    },
    "climate_specific": {
        "sigma_forc": 0.5198,
        "M_pre": 596.4,
        "M": [862.86, 1541.11, 10010.44],
        "Phi": [[0.8240, 0.0767, 0], [0.1760, 0.9183, 0.0007], [0, 0.0050, 0.9993]],
        "sigma": [
            [0.166726698937639, 0.313477283382278],
            [0.0229478090365842, 0.977052190963416],
        ],
    },
}


@pytest.fixture
def setUp():
    return DICE_Prod_Stateupdater(config_dict)


def test_k_t_plus_is_float(setUp):
    assert isinstance(setUp.k_tplus(1, 1, 1, 1, 1, 1, 1, 0.1), float)


def test_m_1_plus_is_float(setUp):
    m2 = np.array([1, 1, 1])
    assert isinstance(setUp.m_1plus(m2, 1), float)


def test_m_2_plus_is_float(setUp):
    m = np.array([1, 1, 1])
    assert isinstance(setUp.m_2plus(m), float)


def test_m_3_plus_is_float(setUp):
    m = np.array([1, 1, 1])
    assert isinstance(setUp.m_3plus(m), float)


def test_tau_1_plus_is_float(setUp):
    tau = np.array([1, 1])
    assert isinstance(setUp.tau_1plus(tau, 1), float)


def test_tau_3_plus_is_float(setUp):
    tau = np.array([1, 1])
    assert isinstance(setUp.tau_1plus(tau, 1), float)
