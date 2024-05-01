import tensorflow as tf
from typing import Tuple
import os

from environments.ace_dice.env_ace_dice import Env_ACE_DICE
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent

from utils.config_loader import load_config
from utils.arg_parsing import parse_model_arguments
from plotting.plot_results import plot_results


def load_network(checkpoint_dir: str, agent: DEQN_agent):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=agent.policy_network)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5
    )

    # Restore the latest checkpoint
    checkpoint_to_restore = checkpoint_manager.latest_checkpoint
    if checkpoint_to_restore:
        checkpoint.restore(checkpoint_to_restore).expect_partial()
        print(f"\nRestoring policy network from {checkpoint_to_restore}")
    else:
        print(f"\nNo trained policy network in path: {checkpoint_dir}")


def initialize(model_version: str) -> Tuple[Env_ACE_DICE, DEQN_agent]:
    base_dir = os.path.dirname(os.path.realpath(__file__))

    env_config_path = os.path.join(
        base_dir, f"configs/state_and_action_space/ace_dice_{model_version}.yaml"
    )
    nw_config_path = os.path.join(
        base_dir, "configs/network_configs/network_config1.yaml"
    )
    checkpoint_dir = os.path.join(base_dir, f"logs/{model_version}/checkpoints")

    env_config = load_config(env_config_path)
    network_config = load_config(nw_config_path)

    environment = Env_ACE_DICE(env_config)

    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    load_network(checkpoint_dir=checkpoint_dir, agent=agent)

    return environment, agent


def main(model_version: str):
    """Responsible for generating the main thesis results.

    Args:
        model_version (str): What version of DICE to show results for. Has to be either 2016 or 2023.
    """
    env, agent = initialize(model_version)

    # Use the plotting script for getting simulated state variables and action variables.
    plot_results(
        env, agent, f"plotting/plots/results/ACE_DICE{model_version}/trajectories"
    )

    # Implement something that generates summary statistics for the objective functions.

    # Implement something that checks the optimal paths by Christian's equations.


if __name__ == "__main__":
    args = parse_model_arguments()
    main(model_version=args.model_version)
