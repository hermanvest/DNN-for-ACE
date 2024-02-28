import numpy as np
import tensorflow as tf

from training_algorithms.algorithm_deqn import Algorithm_DEQN
from agents.deqn_agent import DEQN_agent
from networks.policy_network import Policy_Network
from environments.deqn_ace_dice.ace_dice_env import Ace_dice_env

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
    environment = Ace_dice_env(env_config)

    # Initialization of the agent
    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    # Initialization of algorithm
    algorithm_config["t_max"] = env_config["general"]["t_max"]
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
    print("Generating episode...")
    episodes = algorithm.generate_episodes()

    print("Printing the episode")
    print(episodes[:5].numpy())

    print("================== TERMINATES: test_episode_generation() ==================")


def test_epoch_and_episode():
    print("\n================== RUNNING: test_epoch_and_episode() ==================")
    algorithm = setUp()
    print("Generating episode...")
    episodes = algorithm.generate_episodes()
    print(f"Finished generating an episode with shape: {episodes.numpy().shape}")

    # Creating batches:
    batch_size = 12
    flattened_episodes = tf.reshape(episodes, [-1, episodes.shape[-1]])
    shuffled_episodes = tf.random.shuffle(flattened_episodes)
    batches = tf.data.Dataset.from_tensor_slices(shuffled_episodes).batch(batch_size)
    print(f"Shuffled data with {sum(1 for _ in batches)} batches")

    print("\nRunning an Epoch on the batches...")
    epoch_average_mse = algorithm.epoch(batches)
    print(f"Average MSE over the epoch was: {epoch_average_mse}")

    print("================== TERMINATES: test_epoch_and_episode() ==================")


def test_train_on_episodes():
    print("\n================== RUNNING: test_epoch_and_episode() ==================")
    algorithm = setUp()

    print("Generating episode...")
    episodes = algorithm.generate_episodes()
    print(f"Finished generating an episode with shape: {episodes.numpy().shape}")

    print("training on episodes...")
    algorithm.train_on_episodes(episodes)

    print("================== TERMINATES: test_epoch_and_episode() ==================")


def test_main_training_loop():
    print("\n================== RUNNING: test_epoch_and_episode() ==================")
    algorithm = setUp()

    print("training on episodes...")
    algorithm.main_loop()

    print("================== TERMINATES: test_epoch_and_episode() ==================")


def main():
    print("################## IN MAIN FUNCTION ##################")

    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")

    test_episode_generation()

    print("\n\n#######################################################")
    print("###############     INTEGRATION TESTS    ##############")
    print("#######################################################\n\n")

    test_epoch_and_episode()
    test_train_on_episodes()
    test_main_training_loop()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
