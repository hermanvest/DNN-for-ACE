# Imports
import tensorflow as tf

from environments.ace_dice_2016.ace_dice_2016_env import Ace_dice_2016
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent
from training_algorithms.algorithm_deqn import Algorithm_DEQN
from utils.config_loader import load_config


def main():
    # Paths
    path_to_environment_config = "configs/state_and_action_space/ace_dice_2016.yaml"
    path_to_network_config = "configs/network_configs/network_config1.yaml"
    path_to_algorithm_config = "configs/training_configs/base_configuration.yaml"

    # Loading of config files
    ace_dice_2016_config = load_config(path_to_environment_config)
    network_config = load_config(path_to_network_config)
    algorithm_config = load_config(path_to_algorithm_config)

    # Initialization of environment
    parameter_config = ace_dice_2016_config["parameters"]
    state_configs = ace_dice_2016_config["state_variables"]
    t_max = 10
    num_batches = 10
    env = Ace_dice_2016(t_max, num_batches, ace_dice_2016_config)

    # Initialization of agent and policy network
    network_config["config_env"] = ace_dice_2016_config
    policy_network = Policy_Network(**network_config)
    agent = DEQN_agent(policy_network)

    # Initialization of training algorithm and optimizer
    algorithm_config["env"] = env
    algorithm_config["agent"] = agent
    algorithm_config["optimizer"] = tf.keras.optimizers.Adam(
        learning_rate=0.001, clipvalue=1.0
    )

    algorithm = Algorithm_DEQN(**algorithm_config)

    # Running the algorithm
    algorithm.main_loop()


if __name__ == "__main__":
    main()
