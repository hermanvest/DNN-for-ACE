# pytest -p no:warnings

import pytest
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


@pytest.fixture
def setUp():
    tf.keras.backend.clear_session()

    return Policy_Network(**config)


def test_prediction_shape(setUp):
    # Generate a batch of dummy input data
    dummy_inputs = np.random.rand(
        5, len(env_config["state_variables"])
    )  # Batch size of 5, matching input_space
    model = setUp

    # Perform prediction using the model
    predictions = model.predict(dummy_inputs)

    # Check if the output predictions have the correct shape
    assert predictions.shape == (
        5,
        len(env_config["action_variables"]),
    ), "The shape of the output predictions is incorrect."


def test_x_t_between_zero_and_one(setUp):
    # Generate a batch of dummy input data
    dummy_inputs = np.random.rand(5, len(env_config["state_variables"]))
    model = setUp

    # Perform prediction using the model
    predictions = model.predict(dummy_inputs)

    x_t_predictions = predictions[:, 0]
    assert np.all(
        (x_t_predictions >= 0) & (x_t_predictions <= 1)
    ), f"The predited value for x_t is not between 0 and 1. Prediction was {x_t_predictions}"
