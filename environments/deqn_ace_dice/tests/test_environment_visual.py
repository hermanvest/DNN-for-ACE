import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from pathlib import Path
from environments.deqn_ace_dice.ace_dice_env import Ace_dice_env
from utils.config_loader import load_config
from environments.deqn_ace_dice.computation_utils import custom_sigmoid
from typing import List, Optional

# Define the path to the current script
current_script_path = Path(__file__).parent

# Navigate up to the common root and then to the YAML file
yaml_file_path = (
    current_script_path.parent.parent.parent
    / "configs"
    / "state_and_action_space"
    / "ace_dice_2016.yaml"
)


######################## Uitlity functions ########################
def plot_var(
    title: str,
    xlab: str,
    ylab: str,
    var_name: str,
    plot_directory: str,
    time_steps: List,
    state_variable_values: List,
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
        state_variable_values (List): Values of the primary state variable over time.
        extra_var_name (Optional[str]): Name of the extra variable to be plotted, optional.
        extra_variable_values (Optional[List]): Values of the extra variable over time, optional.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, state_variable_values, marker="o", label=var_name)
    if extra_variable_values is not None:
        plt.plot(time_steps, extra_variable_values, marker="x", label=extra_var_name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    plt.legend()

    # File name for the plot
    plot_filename = f"{var_name}.png"

    # Full path to save the plot
    plot_path = os.path.join(plot_directory, plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory


def get_trajectory(x_t: float, E_t: float, env: Ace_dice_env, t_max: int) -> List:
    s_t = env.reset()  # shape [batch, statevariables]
    states = []  # shape [timestep, statevariables]

    for t in range(t_max):
        # Creating the actions batch, as I expect it to look
        a_t = [x_t, E_t, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Only first two are used
        a_t = tf.convert_to_tensor(a_t)

        action_batch = []
        action_batch.append(a_t)
        action_batch = tf.stack(action_batch)

        s_tplus = env.step(s_t, action_batch)
        states.append(s_t)
        s_t = s_tplus
    return states


def setUp() -> Ace_dice_env:
    config = load_config(yaml_file_path)
    env = Ace_dice_env(config)
    return env


def test_reset_environment() -> None:
    print("\n================== RUNNING: test_reset_environment() ==================")
    env = setUp()
    initial_states = env.reset()
    print(f"\nInitial states: {initial_states.numpy()}")
    print("================== TERMINATES: test_reset_environment() ==================")


def test_step_in_environment_from_state() -> None:
    print(
        "\n================== RUNNING: test_step_in_environment_from_state() =================="
    )
    env = setUp()
    initial_states = env.reset()

    # Creating a sample similar to neural network output for a batch prediction
    action_batch = []
    actions = [0.4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 11 actions
    actions = tf.convert_to_tensor(actions)
    action_batch.append(actions)
    action_batch = tf.stack(action_batch)

    next_state = env.step(initial_states, action_batch)

    print(f"\nInitial states: {initial_states.numpy()}")
    print(f"\nNext state: {next_state.numpy()}")
    print(
        "================== TERMINATES: test_step_in_environment_from_state() =================="
    )


def test_exo_vars() -> None:
    print("\n================== RUNNING: test_exo_vars() ==================")
    env = setUp()

    print("Extracting exo_vars...")
    tfp = env.equations_of_motion.A_t.numpy()
    labor = env.equations_of_motion.N_t.numpy()
    sigma = env.equations_of_motion.sigma.numpy()
    theta_1 = env.equations_of_motion.theta_1.numpy()

    print("Creating plots...")
    # TODO: Plot tfp, labor, sigma and theta_1 in a gridplot and save it to a plot directory
    # Create a figure with subplots in a 2x2 arrangement
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].plot(tfp, "r-")
    axs[0, 0].set_title("A_t")

    axs[0, 1].plot(labor, "g-")
    axs[0, 1].set_title("N_t")

    axs[1, 0].plot(sigma, "b-")
    axs[1, 0].set_title("Sigma")

    axs[1, 1].plot(theta_1, "y-")
    axs[1, 1].set_title("Theta 1")

    # Adjust layout to not overlap
    plt.tight_layout()

    # Directory where you want to save the plot
    plot_directory = "plots/exo_vars"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # File name for the plot
    plot_filename = "exo_vars_plot.png"

    # Full path to save the plot
    plot_path = os.path.join(plot_directory, plot_filename)

    # Save the plot
    plt.savefig(plot_path)

    print("================== TERMINATES: test_exo_vars() ==================")


def test_environment_dynamics() -> None:
    print(
        "\n================== RUNNING: test_environment_dynamics() =================="
    )
    env = setUp()
    t_max = 30
    x_t, E_t = (0.70, 30.0)
    states = get_trajectory(x_t=x_t, E_t=E_t, env=env, t_max=t_max)

    # Plot and save each state variable
    state_names = [item["name"] for item in env.equations_of_motion.states]
    # Save the plot to a file
    plot_directory = "plots/sim_vectorized"
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    time_steps = list(range(t_max))
    for i, name in enumerate(state_names):
        # Extract the ith state variable from each tensor in the states list
        state_variable_values = [state.numpy()[0, i] for state in states]
        plot_var(
            name,
            "Period",
            f"Value of {name}",
            name,
            plot_directory,
            time_steps,
            state_variable_values,
        )

    # Plotting E_t
    capital_path = [state.numpy()[0, 0] for state in states]
    E_t_BAUs = [
        env.equations_of_motion.E_t_BAU(time_steps[i], capital_path[i])
        for i in range(t_max)
    ]
    E_ts = [custom_sigmoid(E_t, E_t_BAUs[i]) for i in range(t_max)]
    plot_var(
        "E_t",
        "Period",
        "Value of E_T",
        "E_t",
        plot_directory,
        time_steps,
        E_ts,
        "E_t_BAU",
        E_t_BAUs,
    )

    # Plotting output Y_t
    Y_gross_simulated = [
        env.equations_of_motion.Y_gross(time_steps[i], capital_path[i])
        for i in range(t_max)
    ]
    plot_var(
        "Y_gross",
        "Period",
        "Value of Y_gross",
        "Y_gross",
        plot_directory,
        time_steps,
        Y_gross_simulated,
    )

    print(
        "================== TERMINATES: test_environment_dynamics() =================="
    )


def main():
    print("################## IN MAIN FUNCTION ##################")

    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")

    test_reset_environment()
    test_step_in_environment_from_state()

    print("\n\n#######################################################")
    print("################## INTEGRATION TESTS ##################")
    print("#######################################################\n\n")

    test_exo_vars()
    test_environment_dynamics()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
