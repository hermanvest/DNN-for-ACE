import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from environments.ace_dice.env_ace_dice import Env_ACE_DICE
from agents.deqn_agent import DEQN_agent
from typing import Tuple, List, Optional
from environments.ace_dice.computation_utils import custom_sigmoid


#################### START HELPER METHODS       ####################
def generate_trajectory(
    env: Env_ACE_DICE, agent: DEQN_agent, t_max: int
) -> Tuple[tf.Tensor, tf.Tensor]:
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


#################### END OF HELPER METHODS      ####################
#################### START RESULT GENERATION    ####################
def store_data(
    states: tf.Tensor,
    actions: tf.Tensor,
    dir: str,
    action_names: List,
    state_names: List,
) -> None:
    """This function takes simulated trajectories for states and actions formatted as they are in the config files of the environments.
    It saves the trajectories to a pandas dataframe that then saves the dataframe as a csv to the specified directory.

    Args:
        states (tf.Tensor): Tensor containing state variables.
        actions (tf.Tensor): Tensor containing action variables.
        dir (str): Directory where the CSV file will be saved.
    """
    print("\n--------------------------------\nNOW GENERATING: Storing trajectory data")

    # Convert Tensors to numpy if not already
    states = states.numpy() if isinstance(states, tf.Tensor) else states
    actions = actions.numpy() if isinstance(actions, tf.Tensor) else actions

    # Create dataframes from the states and actions
    df_states = pd.DataFrame(states, columns=state_names)
    df_actions = pd.DataFrame(actions, columns=action_names)

    # Concatenate the states and actions dataframes
    df = pd.concat([df_states, df_actions], axis=1)

    # Check if directory exists, if not, create it
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Define the full path for the csv file
    file_path = os.path.join(dir, "trajectory_data.csv")

    # Save the dataframe to csv
    df.to_csv(file_path, index=False)

    print(f"Data saved successfully to {file_path}")


def generate_error_statistics(
    states: tf.Tensor,
    actions: tf.Tensor,
    env: Env_ACE_DICE,
    t_max: int,
    dir: str,
) -> None:
    """
    Generate and save a summary statistics table of losses calculated from transitions in an agent-environment interaction over a specified number of time steps. The function calculates losses for each transition based on the provided states and actions, then compiles these into a DataFrame to compute descriptive statistics, and finally outputs these statistics as a LaTeX table saved to a specified directory.

    Args:
        states (tf.Tensor): A tensor containing sequential state information of the environment,
                            where each row represents the state at a particular time step.
        actions (tf.Tensor): A tensor containing sequential action data taken by the agent,
                             where each row corresponds to an action taken at a time step.
        env (Env_ACE_DICE): An instance of the Env_ACE_DICE environment which provides the
                            method `individual_losses_analysis` to compute losses for transitions.
        t_max (int): The total number of time steps to consider for generating statistics.
                     This should be at least 2 to ensure there are transitions to analyze.
        dir (str): The directory path where the LaTeX file containing the summary statistics
                   will be saved. The function will create the directory if it does not exist.

    Returns:
        None: This function does not return any value. It writes a file containing the LaTeX table of
              summary statistics directly to the specified directory.

    Raises:
        FileNotFoundError: If there is an issue accessing or writing to the specified directory.
        Other possible exceptions related to data handling or file I/O may also be raised.

    Example:
        >>> generate_error_statistics(states_tensor, actions_tensor, environment, 100, './results/stats')

    Note:
        This function assumes that the `individual_losses_analysis` method returns a numpy array
        of losses for each transition, and it requires `numpy` and `pandas` libraries for calculations
        and `os` library for file operations. It also assumes TensorFlow tensors are in a format
        compatible with slicing as used in this function.
    """
    print("\n--------------------------------\nNOW GENERATING: Error statistics.")
    # initialize a numpy_array that stores numpy_arrays of losses
    all_losses = []

    # Iterate and calculate and append.
    for t in range(t_max - 1):  # t_max - 1 because we need s_tplus
        s_t = states[t]
        a_t = actions[t]
        s_tplus = states[t + 1]
        a_tplus = actions[t + 1]

        # Assuming individual_losses returns a numpy array of losses for each transition
        losses = env.loss.individual_losses_analysis(s_t, a_t, s_tplus, a_tplus)
        all_losses.append(losses)

    # Generate the table with summary statistics (mean, median, min, max etc.)
    all_losses = np.abs(np.array(all_losses))

    loss_df = pd.DataFrame(
        all_losses, columns=[f"l{i}" for i in range(all_losses.shape[1])]
    )

    summary_statistics = loss_df.describe(percentiles=[0.001, 0.25, 0.5, 0.75, 0.999])

    summary_statistics.drop("count", inplace=True)

    latex_table = summary_statistics.to_latex()

    os.makedirs(dir, exist_ok=True)
    filename = "summary_statistics.tex"
    file_path = os.path.join(dir, filename)
    with open(file_path, "w") as f:
        f.write(latex_table)

    print(f"Error statistics located at: {dir}")


#################### END RESULT GENERATION      ####################
#################### START MAIN METHOD          ####################


def plot_results(env: Env_ACE_DICE, agent: DEQN_agent, plot_dir: str):
    states, actions = generate_trajectory(env, agent, env.equations_of_motion.t_max)

    # Getting state and action names
    state_names = [var["name"] for var in env.equations_of_motion.states]
    action_names = [var["name"] for var in env.equations_of_motion.actions]

    store_data(states, actions, f"{plot_dir}/csv", action_names, state_names)

    generate_error_statistics(
        states,
        actions,
        env,
        env.equations_of_motion.t_max,
        f"{plot_dir}/error_statistics",
    )

    print(
        "\n--------------------------------\nFINISHED: All results have been successfully generated."
    )
