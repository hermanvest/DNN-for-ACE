import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from environments.ace_dice.env_ace_dice import Env_ACE_DICE
from agents.deqn_agent import DEQN_agent
from typing import Tuple, List, Optional
from environments.ace_dice.computation_utils import custom_sigmoid


#################### START CONSTANTS            ####################
def get_sigma_matrix_for_scc(env: Env_ACE_DICE) -> tf.Tensor:
    sigma = env.equations_of_motion.sigma_transition
    ones_matrix = tf.ones_like(sigma)

    # Calculate the inverse of (ones_matrix - beta * sigma)
    matrix_to_invert = ones_matrix - env.equations_of_motion.beta * sigma
    inverse_matrix = tf.linalg.inv(matrix_to_invert)

    # Extract the element at the first row and first column
    result_element = inverse_matrix[0, 0]
    return result_element


def get_Phi_inverse_for_scc(env: Env_ACE_DICE) -> tf.Tensor:
    phi = env.equations_of_motion.Phi
    ones_matrix = tf.ones_like(phi)

    # Calculate the inverse of (ones_matrix - beta * Phi)
    matrix_to_invert = ones_matrix - env.equations_of_motion.beta * phi
    inverse_matrix = tf.linalg.inv(matrix_to_invert)

    # Extract the element at the first row and first column
    result_element = inverse_matrix[0, 0]
    return result_element


def damage_t(env: Env_ACE_DICE, tau_layer_1_t: tf.Tensor) -> tf.Tensor:
    damage_t = 1 - tf.exp(
        -env.equations_of_motion.xi_0 * tau_layer_1_t + env.equations_of_motion.xi_0
    )
    return damage_t


def calculate_scc_t(
    env: Env_ACE_DICE,
    net_output_t: tf.Tensor,
    sigma_inverse: tf.Tensor,
    phi_inverse: tf.Tensor,
) -> tf.Tensor:

    return (
        (
            (tf.pow(env.equations_of_motion.beta, 2) * net_output_t)
            / env.equations_of_motion.M_pre
        )
        * sigma_inverse
        * env.equations_of_motion.sigma_forc
        * phi_inverse
    )


def calculate_optimal_abatement_rate(
    env: Env_ACE_DICE, tau_layer_1_t: tf.Tensor, scc_t: tf.Tensor, t: int
) -> tf.Tensor:
    p_back_t = env.equations_of_motion.pbacktime[t]
    fraction = scc_t / (p_back_t * (1 - damage_t(env, tau_layer_1_t)))

    exponent = 1 / (env.equations_of_motion.theta_2 - 1)
    return tf.pow(fraction, exponent)


#################### END CONSTANTS              ####################
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


def get_adj_emissions_and_bau_emissions(
    env: Env_ACE_DICE,
    capital_path: tf.Tensor,
    emissions_path: tf.Tensor,
    time_steps: List,
) -> Tuple[np.array, np.array]:

    emissions_BAU_list = []
    emissions_adjusted_list = []

    for t in time_steps:
        k_t = capital_path[t]
        E_t = emissions_path[t]

        E_t_BAU = env.equations_of_motion.E_t_BAU(t, k_t)
        emissions_BAU_list.append(E_t_BAU.numpy())

        E_t_adjusted = custom_sigmoid(E_t, E_t_BAU).numpy()
        emissions_adjusted_list.append(E_t_adjusted)

    # Convert lists to numpy arrays before returning
    emissions_BAU_np = np.array(emissions_BAU_list)
    emissions_adjusted_np = np.array(emissions_adjusted_list)

    return emissions_adjusted_np, emissions_BAU_np


#################### END OF HELPER METHODS      ####################
#################### START RESULT GENERATION    ####################


def plot_states(
    states: tf.Tensor, years: List, plot_dir: str, env: Env_ACE_DICE
) -> None:
    print(
        "\n--------------------------------\nNOW GENERATING: Results for state variables."
    )
    os.makedirs(plot_dir, exist_ok=True)

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
    print(f"Plots located at {plot_dir}")


def plot_actions(
    env: Env_ACE_DICE,
    actions: tf.Tensor,
    states: tf.Tensor,
    action_names: List,
    time_steps: List,
    years: List,
    plot_dir: str,
) -> None:
    print(
        "\n--------------------------------\nNOW GENERATING: Results for action variables."
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Plotting x_t
    consumption_rate_path = actions[:, 0].numpy() * 100  # Converting to percent
    plt.plot(years, consumption_rate_path, label="Consumption Rate")
    plt.xlabel("Years")
    plt.ylabel("Rate (%)")
    filename = "consumption_rate_plot.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    # Plotting E_t and BAU
    emissions_path = actions[:, 1]
    capital_path = states[:, 0]
    emissions_adjusted, emissions_BAU = get_adj_emissions_and_bau_emissions(
        env, capital_path, emissions_path, time_steps
    )
    plt.plot(years, emissions_adjusted, label="Emissions After Abatement")
    plt.plot(years, emissions_BAU, label="BAU Emissions")
    plt.xlabel("Years")
    plt.ylabel("GtCO2")
    filename = "emissions_plot.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    # Loop through the rest and generate the plots
    start_index = 2
    for variable_index, name in enumerate(
        action_names[start_index:], start=start_index
    ):
        action_data = actions[:, variable_index].numpy()
        plt.plot(years, action_data, label=f"{name}")
        plt.xlabel("Years")
        plt.ylabel(f"Shadow value of {name}")
        filename = f"{name}_plot.png"
        file_path = os.path.join(plot_dir, filename)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

    print(f"Plots located at {plot_dir}")


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


def scc_and_optimal_abatement_path(
    env: Env_ACE_DICE,
    states: tf.Tensor,
    actions: tf.Tensor,
    t_max: int,
    years: List,
    plot_dir: str,
):
    print(
        "\n--------------------------------\nNOW GENERATING: Analytic derivations (SCC and abatement rate)."
    )
    os.makedirs(plot_dir, exist_ok=True)
    sigma_inverse = get_sigma_matrix_for_scc(env)
    phi_inverse = get_Phi_inverse_for_scc(env)
    emissions_path = actions[:, 1]
    capital_path = states[:, 0]
    tau_layer1 = states[:, 4]
    scc_path = []
    optimal_abatement_rate_path = []

    for t in range(t_max):
        E_t = emissions_path[t]
        k_t = capital_path[t]

        net_output_t = tf.exp(env.equations_of_motion.log_Y_t(k_t, E_t, t)) * (
            1 - damage_t(env, tau_layer1[t])
        )
        scc_t = calculate_scc_t(env, net_output_t, sigma_inverse, phi_inverse)
        scc_path.append(scc_t)
        optimal_abatement_rate_path.append(
            calculate_optimal_abatement_rate(env, tau_layer1[t], scc_t, t)
        )

    # Plotting SCC
    scc_path_np = np.array(scc_path)
    plt.plot(years, scc_path_np, label="Social Cost of Carbon")
    plt.xlabel("Years")
    plt.ylabel("SCC in utils")
    filename = "SCC.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    # Plotting Optimal Anatement Rate
    optimal_abatement_rate_path_np = np.array(optimal_abatement_rate_path)
    plt.plot(years, optimal_abatement_rate_path_np, label="Optimal Abatement Rate Path")
    plt.xlabel("Years")
    plt.ylabel("Rate (decimal)")
    filename = "optimal_abatement_rate_path.png"
    file_path = os.path.join(plot_dir, filename)
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    print(f"Analytic derivations located at: {plot_dir}")


#################### END RESULT GENERATION      ####################
#################### START MAIN METHOD          ####################


def plot_results(env: Env_ACE_DICE, agent: DEQN_agent, plot_dir: str):
    states, actions = generate_trajectory(env, agent, env.equations_of_motion.t_max)
    time_steps = list(range(env.equations_of_motion.t_max))
    years = [x * 5 for x in time_steps]

    # Getting state and action names
    action_names = [var["name"] for var in env.equations_of_motion.actions]

    plot_states(states, years, f"{plot_dir}/states", env)
    plot_actions(
        env, actions, states, action_names, time_steps, years, f"{plot_dir}/actions"
    )

    generate_error_statistics(
        states,
        actions,
        env,
        env.equations_of_motion.t_max,
        f"{plot_dir}/error_statistics",
    )

    scc_and_optimal_abatement_path(
        env,
        states,
        actions,
        env.equations_of_motion.t_max,
        years,
        f"{plot_dir}/analytic_derivations",
    )

    print(
        "\n--------------------------------\nFINISHED: All results have been successfully generated."
    )
