import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


from environments.ace_dice.ace_dice_env import Ace_dice_env
from networks.policy_network import Policy_Network
from agents.deqn_agent import DEQN_agent
from utils.config_loader import load_config
from typing import Tuple, List


def setup() -> Tuple[Ace_dice_env, DEQN_agent, List, List, int]:
    """
    Sets up the environment, agent, and extracts state and action variable names
    from configuration files for the Ace Dice 2016 simulation.

    This function performs the following steps:
    1. Loads configuration settings from YAML files for the environment, network, and algorithm.
    2. Extracts names of state and action variables from the environment configuration.
    3. Initializes the Ace_dice_2016 environment with the loaded configuration.
    4. Initializes the policy network and the DEQN agent with the network and environment configurations.
    5. Returns the initialized environment, agent, lists of state and action variable names, and the maximum number of timesteps (t_max) from the configuration.

    Returns:
        Tuple[Ace_dice_2016, DEQN_agent, List[str], List[str], int]: A tuple containing:
        - The initialized Ace_dice_2016 environment instance.
        - The initialized DEQN_agent instance.
        - A list of strings representing the names of state variables.
        - A list of strings representing the names of action variables.
        - An integer representing the maximum number of timesteps (t_max) for the simulation, extracted from the environment configuration.

    Note:
    - Configuration paths for the environment, network, and algorithm are hardcoded within the function.
    - This function assumes the presence of 'state_variables' and 'action_variables' keys within the environment configuration file,
      and a 'general' key with a 't_max' subkey indicating the maximum number of timesteps.
    """
    # Paths
    env_config_path = "configs/state_and_action_space/ace_dice_2016.yaml"
    nw_config_path = "configs/network_configs/network_config1.yaml"
    algorithm_config_path = "configs/training_configs/base_configuration.yaml"

    # Loading configs
    env_config = load_config(env_config_path)
    network_config = load_config(nw_config_path)
    algorithm_config = load_config(algorithm_config_path)

    # Getting state and action names
    state_names = [var["name"] for var in env_config["state_variables"]]
    action_names = [var["name"] for var in env_config["action_variables"]]

    # Initialization of the environment
    environment = Ace_dice_env(env_config)

    # Initialization of the agent
    network_config["config_env"] = env_config
    network = Policy_Network(**network_config)
    agent = DEQN_agent(network)

    return environment, agent, state_names, action_names, env_config["general"]["t_max"]


def simulate_episodes_with_trained_agent(
    env: Ace_dice_env, agent: DEQN_agent, t_max: int
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


def clear_plot_directory(plot_dir: str) -> None:
    """
    Removes all files in the specified plot directory.

    Args:
        plot_dir (str): The path to the plot directory to be cleared.
    """
    # Check if the directory exists
    if os.path.exists(plot_dir):
        # Remove all files and subdirectories in the plot directory
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and all its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # If the directory does not exist, create it
        os.makedirs(plot_dir, exist_ok=True)


def generate_plots(
    simulated_states: tf.Tensor,
    simulated_actions: tf.Tensor,
    state_names: List,
    action_names: List,
    env: Ace_dice_env,
    plot_dir: str = "plots/ace_dice_2016",
) -> None:
    # Ensure the plot directory exists
    clear_plot_directory(plot_dir)

    # Assuming name index in name lists are corresponding to index in state and action tensors
    # Loop through the states and generate the relevant plots
    for timestep, name in enumerate(state_names):
        state_data = simulated_states[:, timestep].numpy()
        plt.figure()
        plt.plot(state_data, label=f"{name}")
        plt.title(f"State Variable: {name}")
        plt.xlabel("Time Step")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"simulated_{name}.png"))
        plt.close()

    # Loop through the actions and generate the relevant plots
    for timestep, name in enumerate(action_names):
        action_data = simulated_actions[:, timestep].numpy()
        plt.figure()
        plt.plot(action_data, label=f"{name}")
        plt.title(f"Action Variable: {name}")
        plt.xlabel("Time Step")
        plt.ylabel(name)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"simulated_{name}_action.png"))
        plt.close()

    # Comparison of Emissions from policy network and adjusted ones used in environment
    capital = simulated_states[:, 0]

    # Generating a list of E_t_BAU along simulated path
    emissions_BAU = np.array(
        [
            env.equations_of_motion.E_t_BAU(timestep, capital[timestep]).numpy()
            for timestep in range(len(capital))
        ]
    )
    emissions = simulated_actions[:, 1].numpy()
    emissions_adj = np.where(emissions > emissions_BAU, emissions_BAU, emissions)

    # Plotting the comparison
    plt.figure()
    plt.plot(emissions, label="Emissions (unadjusted)")
    plt.plot(emissions_BAU, label="Emissions BAU", linestyle="--")
    plt.plot(emissions_adj, label="Emissions (adjusted)", linestyle=":")
    plt.title("Emissions Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Emissions")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "emissions_comparison.png"))
    plt.close()


def main():
    print("##################### INITIALIZING ENV AND AGENT  #####################")
    environment, agent, state_names, action_names, t_max = setup()

    print("##################### RESTORING NETWORK WEIGHTS   #####################")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipvalue=1.0)

    checkpoint_dir = "checkpoints/ace_dice_2016"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=agent.policy_network)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=5
    )

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        print(f"\nRestoring policynetwork from {checkpoint_manager.latest_checkpoint}")
    else:
        print("\nNo previous checkpoint for policynetwork.")
        exit(1)

    print("##################### SIMULATING EPISODE WITH NW  #####################")
    simulated_states, simulated_actions = simulate_episodes_with_trained_agent(
        environment, agent, t_max
    )

    generate_plots(
        simulated_states, simulated_actions, state_names, action_names, environment
    )

    print("\n##################### FINISHED GENERATING PLOTS   #####################")
    print(f"Look in the plots directory for results.")


if __name__ == "__main__":
    main()
