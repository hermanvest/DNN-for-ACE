import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from environments.deqn_ace_dice.env_ace_dice import Env_ACE_DICE
from agents.deqn_agent import DEQN_agent
from typing import Tuple, List, Optional
from environments.deqn_ace_dice.computation_utils import custom_sigmoid


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
    states: tf.Tensor,
    state_names: List,
    time_steps: List,
    plot_dir: str,
) -> None:
    # Assuming name index in name lists are corresponding to index in state and action tensors
    # Loop through the states and generate the relevant plots
    for variable_index, name in enumerate(state_names):
        state_data = states[:, variable_index].numpy()
        plot_var(
            f"State: {name}",
            "Time step",
            f"Value of {name}",
            name,
            plot_dir,
            time_steps,
            state_data,
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


def plot_ACE_DICE_trajectories(env: Env_ACE_DICE, agent: DEQN_agent, plot_dir: str):
    states, actions = generate_trajectory(env, agent, env.equations_of_motion.t_max)
    time_steps = list(range(env.equations_of_motion.t_max))

    # Getting state and action names
    state_names = [var["name"] for var in env.equations_of_motion.states]
    action_names = [var["name"] for var in env.equations_of_motion.actions]

    plot_states(states, state_names, time_steps, f"{plot_dir}/states")
    plot_actions(actions, action_names, time_steps, f"{plot_dir}/actions")

    emissions_path = actions[:, 1]
    capital_path = states[:, 0]
    plot_E_t(env, emissions_path, capital_path, time_steps, f"{plot_dir}/actions")
