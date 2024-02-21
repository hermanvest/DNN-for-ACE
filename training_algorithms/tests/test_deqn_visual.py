import numpy as np
import tensorflow as tf

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

algorithm_config_path = (
    current_script_path.parent.parent
    / "configs"
    / "training_configs"
    / "base_configuration.yaml"
)


def setUp() -> Algorithm_DEQN:
    # Loading configs
    env_config = load_config(env_config_path)
    network_config = load_config(nw_config_path)
    algorithm_config = load_config(algorithm_config_path)

    # Initialization of the environment
    environment = Ace_dice_2016(env_config)

    # Initialization of the agent
    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    # Initialization of algorithm
    algorithm_config["env"] = environment
    algorithm_config["agent"] = agent
    algorithm_config["optimizer"] = tf.keras.optimizers.Adam(
        learning_rate=0.001, clipvalue=1.0
    )

    algorithm = Algorithm_DEQN(**algorithm_config)

    return algorithm


def test_episode_generation():
    print("\n================== RUNNING: test_episode_generation() ==================")
    algorithm = setUp()
    episodes = algorithm.generate_episodes().numpy()

    counter = 0
    for timestep in episodes:
        print(f"State variables at time {counter}: {timestep}")
    print("================== TERMINATES: test_episode_generation() ==================")


def main():
    print("################## IN MAIN FUNCTION ##################")

    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")
    test_episode_generation()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
