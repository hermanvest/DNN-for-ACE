# Imports
import tensorflow as tf

from environments.deqn_ace_dice.env_ace_dice import Env_ACE_DICE
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent
from training_algorithms.algorithm_deqn import Algorithm_DEQN
from utils.config_loader import load_config


def main():
    # Paths
    env_config_path = "configs/state_and_action_space/ace_dice_2016.yaml"
    nw_config_path = "configs/network_configs/network_config1.yaml"
    algorithm_config_path = "configs/training_configs/base_configuration.yaml"

    # Loading configs
    env_config = load_config(env_config_path)
    network_config = load_config(nw_config_path)
    algorithm_config = load_config(algorithm_config_path)

    # Initialization of the environment
    environment = Env_ACE_DICE(env_config)

    # Initialization of the agent
    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    # Initialization of algorithm
    algorithm_config["t_max"] = env_config["general"]["t_max"]
    algorithm_config["env"] = environment
    algorithm_config["agent"] = agent
    algorithm_config["optimizer"] = tf.keras.optimizers.Adam(
        learning_rate=1e-5, clipvalue=1.0
    )
    algorithm_config["checkpoint_dir"] = "checkpoints/ace_dice_2016"

    algorithm = Algorithm_DEQN(**algorithm_config)

    # Running the algorithm
    algorithm.main_loop()


if __name__ == "__main__":
    main()
