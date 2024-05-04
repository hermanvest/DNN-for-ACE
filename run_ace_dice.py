import tensorflow as tf

from environments.ace_dice.env_ace_dice import Env_ACE_DICE
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent
from training_algorithms.algorithm_deqn import Algorithm_DEQN
from plotting.plot_results import plot_results
from utils.config_loader import load_config
from utils.arg_parsing import parse_model_arguments


def main(model_version: str):
    # Paths
    # parse the model version
    env_config_path = f"configs/state_and_action_space/ace_dice_{model_version}.yaml"
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
    learning_rate = 1e-4
    algorithm_config["t_max"] = env_config["general"]["t_max"]
    algorithm_config["env"] = environment
    algorithm_config["agent"] = agent
    algorithm_config["optimizer"] = tf.keras.optimizers.Adam(
        learning_rate=learning_rate, clipvalue=1.0
    )
    algorithm_config["log_dir"] = f"logs/{model_version}/training_stats"
    algorithm_config["checkpoint_dir"] = f"logs/{model_version}/checkpoints"

    algorithm = Algorithm_DEQN(**algorithm_config)
    # algorithm.optimizer = tf.keras.optimizers.Adam(
    #    learning_rate=learning_rate,  # clipvalue=1.0
    # )

    # Running the algorithm
    algorithm.main_loop()

    # Plotting the trained model
    plot_results(
        environment, agent, f"plotting/plots/model_run/ACE_DICE{model_version}"
    )


if __name__ == "__main__":
    args = parse_model_arguments()
    main(model_version=args.model_version)
