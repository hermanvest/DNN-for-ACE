import numpy as np
from pathlib import Path
import tensorflow as tf

from environments.ace_dice.ace_dice_env import Ace_dice_env
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


def main():
    print("################## IN MAIN FUNCTION ##################")

    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")

    test_reset_environment()
    test_step_in_environment_from_state()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
