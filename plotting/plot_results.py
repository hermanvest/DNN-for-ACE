import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from environments.ace_dice.env_ace_dice import Env_ACE_DICE
from agents.deqn_agent import DEQN_agent
from typing import Tuple, List, Optional
from environments.ace_dice.computation_utils import custom_sigmoid


def plot_var(
    title: str,
    xlab: str,
    ylab: str,
    var_name: str,
    plot_directory: str,
    time_steps: List,
    variable_values: List,
    extra_var_name: Optional[str] = None,
    extra_variable_values: Optional[List] = None,
) -> None:
    """Saves plot from given arguments including an optional extra variable

    Args:
        title (str): Title of the plot.
        xlab (str): Label for the X-axis.
        ylab (str): Label for the Y-axis.
        var_name (str): Name of the primary variable to be plotted.
        plot_directory (str): Directory path to save the plot.
        time_steps (List): Time steps for the variables.
        variable_values (List): Values of the variable over time.
        extra_var_name (Optional[str]): Name of the extra variable to be plotted, optional.
        extra_variable_values (Optional[List]): Values of the extra variable over time, optional.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, variable_values, marker="o", label=var_name)
    if extra_variable_values is not None:
        plt.plot(time_steps, extra_variable_values, marker="x", label=extra_var_name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.legend()

    # Ensure the plot directory exists
    os.makedirs(plot_directory, exist_ok=True)

    # File name for the plot
    plot_filename = f"{var_name}.png"

    # Full path to save the plot
    plot_path = os.path.join(plot_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory


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


def plot_states(
    states: tf.Tensor, years: List, plot_dir: str, env: Env_ACE_DICE
) -> None:
    # Plotting of capital
    aggregate_capital_path = tf.math.exp(states[:, 0]).numpy()
    plt.plot(years, aggregate_capital_path, label="Aggregate Capital")

    plt.xlabel("Years")
    plt.ylabel("Trillion 2010 usd")

    filename = "aggregate_capital_plot.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")

    plt.close()

    # Plotting of carbon reservoirs
    carbon_reservoirs = states[:, 1:4].numpy()
    plt.plot(years, carbon_reservoirs[:, 0], label="Reservoir 1")
    plt.plot(years, carbon_reservoirs[:, 1], label="Reservoir 2")
    plt.plot(years, carbon_reservoirs[:, 2], label="Reservoir 3")

    plt.xlabel("Years")
    plt.ylabel("GtCO2")
    plt.legend()

    filename = "carbon_reservoirs_plot.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")

    plt.close()

    # Plotting of temperatures
    temperature_layer = (
        tf.math.log(states[:, 4:6]).numpy() / env.equations_of_motion.xi_0
    )
    plt.plot(years, temperature_layer[:, 0], label="Layer 1")
    plt.plot(years, temperature_layer[:, 1], label="Layer 2")

    plt.xlabel("Years")
    plt.ylabel("Deg. C.")
    plt.legend()

    filename = "temperatures_plot.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")

    plt.close()


def plot_E_t(
    env: Env_ACE_DICE,
    emissions_path: tf.Tensor,
    capital_path: tf.Tensor,
    time_steps: List,
    plot_dir: str,
) -> None:
    emissions_BAU = np.array(
        [
            env.equations_of_motion.E_t_BAU(timestep, capital_path[timestep]).numpy()
            for timestep in range(len(capital_path))
        ]
    )
    emissions_adjusted = [
        custom_sigmoid(emissions_path[i], emissions_BAU[i]) for i in time_steps
    ]
    plot_var(
        "Action: E_t",
        "Time step",
        "Value of E_T",
        "E_t",
        plot_dir,
        time_steps,
        emissions_adjusted,
        "E_t_BAU",
        emissions_BAU,
    )


def plot_actions(
    actions: tf.Tensor,
    action_names: List,
    time_steps: List,
    plot_dir: str,
) -> None:
    # Loop through the actions and generate the relevant plots
    for variable_index, name in enumerate(action_names):
        if name == "E_t":
            continue
        action_data = actions[:, variable_index].numpy()
        plot_var(
            f"Action: {name}",
            "Time step",
            f"Value of {name}",
            name,
            plot_dir,
            time_steps,
            action_data,
        )


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


def plot_results(env: Env_ACE_DICE, agent: DEQN_agent, plot_dir: str):
    states, actions = generate_trajectory(env, agent, env.equations_of_motion.t_max)
    time_steps = list(range(env.equations_of_motion.t_max))
    years = [x * 5 for x in time_steps]

    # Getting state and action names
    action_names = [var["name"] for var in env.equations_of_motion.actions]

    plot_states(states, years, f"{plot_dir}/states", env)
    plot_actions(actions, action_names, time_steps, f"{plot_dir}/actions")

    emissions_path = actions[:, 1]
    capital_path = states[:, 0]
    plot_E_t(env, emissions_path, capital_path, time_steps, f"{plot_dir}/actions")

    generate_error_statistics(
        states,
        actions,
        agent,
        env,
        env.equations_of_motion.t_max,
        f"{plot_dir}/error_statistics",
    )
