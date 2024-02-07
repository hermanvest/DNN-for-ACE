import pytest
import numpy as np
import tensorflow as tf

from networks.policy_network import Policy_Network

network_config = {
    "input_space": 10,
    "hidden_nodes": 1024,
    "hidden_activation_function": "relu",
    "output_activation_function": "linear",
    "output_space": 10,
    "initializer_mode": "fan_avg",
    "initializer_distribution": "uniform",
    "initializer_scale": 1.0,
    "initializer_seed": 1,
}


@pytest.fixture
def setUp():
    tf.keras.backend.clear_session()
    return Policy_Network(network_config)


def test_prediction(setUp):
    # Generate a batch of dummy input data
    dummy_inputs = np.random.rand(
        5, network_config["input_space"]
    )  # Batch size of 5, matching input_space
    # Use the setUp fixture to get an instance of your Policy_Network
    model = setUp

    # Perform prediction using the model
    predictions = model.predict(dummy_inputs)

    # Check if the output predictions have the correct shape
    assert predictions.shape == (
        5,
        network_config["output_space"],
    ), "The shape of the output predictions is incorrect."
