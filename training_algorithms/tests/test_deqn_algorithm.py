import pytest
import numpy as np

from training_algorithms.algorithm_deqn import Algorithm_DEQN
from agents.deqn_agent import DEQN_agent
from networks.policy_network import Policy_Network
from environments.ace_dice_2016.ace_dice_2016_env import Ace_dice_2016


network_config = {
    "input_space": 10,
    "hidden_nodes": 1024,
    "hidden_activation_function": "relu",
    "output_activation_function": "linear",
    "output_space": 10,
    "initializer_mode": "fan_avg",
    "initializer_distribution": "uniform",
    "initializer_scale": 1.0,
    "initializer_seed": 1,
}

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
    environment = Ace_dice_2016(env_config)

    return Algorithm_DEQN(agent=agent, env=environment)


"""

def test_that_episodes_are_generated(setUp):
    num_trantitions_expected = 100
    episode = setUp.generate_episode()
    episode_legth = episode.shape[0]
    assert episode == num_trantitions_expected


def test_network_parameters_updates(setUp):
    pre_training_weights = setUp.agent.policy_network.get_weights()
    setUp.do_learning_pass()
    post_training_weights = setUp.agent.policy_network.get_weights()

    updated = False
    for initial, updated in zip(pre_training_weights, post_training_weights):
        if not np.array_equal(initial, updated):
            updated = True
            break

    assert updated
"""
