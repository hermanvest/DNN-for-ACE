import tensorflow as tf
import matplotlib.pyplot as plt
import os

from environments.ace_dice_2016.ace_dice_2016_env import Ace_dice_2016
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent
from utils.config_loader import load_config
from typing import Tuple


def setup() -> Tuple[Ace_dice_2016, DEQN_agent, int]:
    # Paths
    env_config_path = "configs/state_and_action_space/ace_dice_2016.yaml"
    nw_config_path = "configs/network_configs/network_config1.yaml"
    algorithm_config_path = "configs/training_configs/base_configuration.yaml"

    # Loading configs
    env_config = load_config(env_config_path)
    network_config = load_config(nw_config_path)
    algorithm_config = load_config(algorithm_config_path)

    # Only need to simulate one batch for plotting
    env_config["general"]["num_batches"] = 1

    # Initialization of the environment
    environment = Ace_dice_2016(env_config)

    # Initialization of the agent
    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    return environment, agent, env_config["general"]["t_max"]


def load_model(
    checkpoint_dir: str, optimizer: tf.keras.optimizers.Optimizer, model: tf.keras.Model
):
    """
    Attempts to restore a model and its optimizer from the most recent checkpoint in the specified directory.
    Prints a message indicating whether the restoration was successful or if no checkpoint was found.

    Args:
        checkpoint_dir (str): The directory where model checkpoints are saved.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer associated with the model.
        model (tf.keras.Model): The model to be restored.

    Returns:
        tf.keras.Model: The restored model if a checkpoint was found and successfully restored; otherwise, None.
    """
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5
    )

    # TODO: Restore model from previous checkpoint. If no previous checkpoint, return None
    if checkpoint_manager.latest_checkpoint:
        restoration_status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
        restoration_status.assert_consumed()
        print(f"\nRestoring policynetwork from {checkpoint_manager.latest_checkpoint}")
        return model
    else:
        print("\nNo previous checkpoint for policynetwork.")
        return None


def simulate_episodes_with_trained_agent(
    env: Ace_dice_2016, agent: DEQN_agent, t_max: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generates an episode with the policy network and returns a tuple consisting of a tensor with states visited
    and actions taken. The function generates one episode of maximum length `t_max`, collecting states and actions
    at each timestep.

    Args:
        env: The environment instance from which episodes are generated.
        agent: The trained agent that provides the policy for action selection.
        t_max (int): The maximum number of timesteps to simulate in each episode.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing two tensors. The first tensor contains the states visited,
        reshaped to [timesteps, state variables]. The second tensor contains the actions taken, reshaped to
        [timesteps, action variables].
    """

    s_t = env.reset()  # shape: [1, statevariables]
    states = []  # shape: [timestep, 1, statevariables]
    actions = []  # shape: [timestep, batchnumber, actionvariables]

    for _ in range(t_max):
        a_t = agent.get_actions(s_t)  # shape [batch, actionvars]
        s_tplus = env.step(s_t, a_t)

        # Removing batch dimension with squeeze as the batch dimension is 1
        states.append(tf.squeeze(s_t))
        actions.append(tf.squeeze(a_t))

        s_t = s_tplus

    states_tensor = tf.stack(states)  # shape will be [timesteps, statevariables]
    actions_tensor = tf.stack(actions)  # shape will be [timesteps, actionvariables]

    return states_tensor, actions_tensor


def generate_plots(
    simualted_states: tf.Tensor,
    simulated_actions: tf.Tensor,
    plot_dir: str = "plots/ace_dice_2016/state_variables",
):
    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    capital_data = simualted_states[:, 0].numpy()
    plt.figure()
    plt.plot(capital_data, label=f"Simulated path of Capital")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"simulated_capital.png"))
    plt.close()


def main():
    environment, agent, t_max = setup()

    # Loading model
    checkpoint_dir = "checkpoints/ace_dice_2016"
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

    loaded_model = load_model(checkpoint_dir, agent.policy_network, optimizer)
    if loaded_model is not None:
        agent.policy_network = loaded_model
    else:
        # Failed to load the model from weights. Exiting plotting program
        exit(1)

    # Simulating an episode with trained network
    simulated_states, simulated_actions = simulate_episodes_with_trained_agent(
        environment, agent, t_max
    )

    generate_plots(simulated_states, simulated_actions)


if __name__ == "__main__":
    main()
