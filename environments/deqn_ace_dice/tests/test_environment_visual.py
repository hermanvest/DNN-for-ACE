import numpy as np
from pathlib import Path
import tensorflow as tf

from environments.deqn_ace_dice.ace_dice_env import Ace_dice_env
from utils.config_loader import load_config

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
def plot_trajectory():
    pass


def setUp() -> Ace_dice_env:
    config = load_config(yaml_file_path)
    env = Ace_dice_env(config)
    return env


def test_reset_environment():
    print("\n================== RUNNING: test_reset_environment() ==================")
    env = setUp()
    initial_states = env.reset()
    print(f"\nInitial states: {initial_states.numpy()}")
    print("================== TERMINATES: test_reset_environment() ==================")


def test_step_in_environment_from_state():
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


def test_environment_dynamics():
    print(
        "\n================== RUNNING: test_environment_dynamics() =================="
    )
    env = setUp()
    s_t = env.reset()  # shape [batch, statevariables]
    states = []  # shape [timestep, statevariables]

    for t in range(10):
        # Getting a resonable value for E_t
        k_t = s_t[0, 0]
        E_t = 100000  # To get it close to E_t_BAU
        x_t = 0.75
        print(f"k_t: {k_t}, E_t: {E_t}")

        # Creating the actions batch, as I expect it to look
        a_t = [x_t, E_t, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Only first two are used
        a_t = tf.convert_to_tensor(a_t)

        action_batch = []
        action_batch.append(a_t)
        action_batch = tf.stack(action_batch)

        s_tplus = env.step(s_t, action_batch)
        states.append(s_t)
        s_t = s_tplus

    print(states)
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

    test_environment_dynamics()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
