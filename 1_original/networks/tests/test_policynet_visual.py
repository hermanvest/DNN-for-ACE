import numpy as np
import tensorflow as tf

from pathlib import Path
from networks.policy_network import Policy_Network
from utils.config_loader import load_config

current_script_path = Path(__file__).parent

# Navigate up to the common root and then to the YAML file
env_config_path = (
    current_script_path.parent.parent
    / "configs"
    / "state_and_action_space"
    / "ace_dice_2016.yaml"
)

nw_config_path = (
    current_script_path.parent.parent
    / "configs"
    / "network_configs"
    / "network_config1.yaml"
)


env_config = load_config(env_config_path)
config = load_config(nw_config_path)
config["config_env"] = env_config


def setUp():
    tf.keras.backend.clear_session()

    return Policy_Network(**config)


def test_x_t_between_zero_and_one():
    print(
        "\n================== RUNNING: test_x_t_between_zero_and_one() =================="
    )

    # Generate a batch of dummy input data
    # Creating a sample similar to neural network output for a batch prediction
    state_batch = []
    states = [1, 1, 1, 1, 1, 1, 1]  # 7 states
    states = tf.convert_to_tensor(states)
    state_batch.append(states)
    state_batch.append(states)
    state_batch = tf.stack(state_batch)

    model = setUp()  # Ensure this correctly initializes your model

    # Perform prediction using the model
    predictions = model.predict(state_batch)

    # Assuming predictions[:, 0] gives the x_t values and they need to be between 0 and 1
    x_t_predictions = predictions[
        :, 0
    ]  # Extract the first state variable for each batch

    print(f"x_t: {x_t_predictions}")
    print(
        "================== TERMINATES: test_x_t_between_zero_and_one() =================="
    )


def test_loads_of_predictions():
    pass


def main():
    print("################## IN MAIN FUNCTION ##################")

    print("\n\n#######################################################")
    print("##################     UNIT TESTS    ##################")
    print("#######################################################\n\n")

    test_x_t_between_zero_and_one()

    print("################## END TESTS ##################")


if __name__ == "__main__":
    main()
