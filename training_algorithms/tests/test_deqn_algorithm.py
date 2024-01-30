import pytest

from training_algorithms.algorithm_deqn import Algorithm_DEQN
from agents.deqn_agent import DEQN_agent
from networks.policy_network import Policy_Network
from environments.base_ace_with_dice_prod.base_ace_dice_prod import Base_ace_dice_prod


network_config = {"input_space": 10, "output_space": 10, "hidden_nodes": 1024}

env_config = {
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
    network = Policy_Network(network_config)
    agent = DEQN_agent(network)
    environment = Base_ace_dice_prod(env_config)

    return Algorithm_DEQN(agent=agent, env=environment)
