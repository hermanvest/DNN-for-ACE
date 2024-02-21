import pytest
import numpy as np

from training_algorithms.algorithm_deqn import Algorithm_DEQN
from agents.deqn_agent import DEQN_agent
from networks.policy_network import Policy_Network
from environments.ace_dice_2016.ace_dice_2016_env import Ace_dice_2016

from pathlib import Path
from utils.config_loader import load_config

current_script_path = Path(__file__).parent

# Navigate up to the common root and then to the YAML file
env_config_path = (
    current_script_path.parent.parent
    / "configs"
    / "state_and_action_space"
    / "ace_dice_2016.yaml"
)

nw_config_path = (
    current_script_path.parent.parent
    / "configs"
    / "network_configs"
    / "network_config1.yaml"
)

algorithm_config = (
    current_script_path.parent.parent
    / "configs"
    / "network_configs"
    / "network_config1.yaml"
)


env_config = load_config(env_config_path)
network_config = load_config(nw_config_path)
network_config["config_env"] = env_config


@pytest.fixture
def setUp():
    network = Policy_Network(network_config)
    agent = DEQN_agent(network)
    environment = Ace_dice_2016(env_config)

    return Algorithm_DEQN(agent=agent, env=environment)
