import pytest
import numpy as np
import tensorflow as tf

from networks.policy_network import Policy_Network

network_config = {
    "hidden_nodes": 1024,
    "hidden_activation_function": "relu",
    "output_activation_function": "linear",
    "kernel_initializer_config": {
        "mode": "fan_avg",
        "distribution": "uniform",
        "scale": 1.0,
        "seed": 1,
    },
}

env_config = {
    "state_variables": [
        {"name": "k_t"},
        {"name": "M_1_t"},
        {"name": "M_2_t"},
        {"name": "M_3_t"},
        {"name": "tau_1_t"},
        {"name": "tau_2_t"},
        {"name": "t"},
    ],
    "action_variables": [
        {"name": "x_t", "activation": "tf.keras.activations.linear"},
        {"name": "E_t", "activation": "tf.keras.activations.linear"},
        {"name": "V_t", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_k_t", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_m_1", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_m_2", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_m_3", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_tau_1", "activation": "tf.keras.activations.linear"},
        {"name": "lambda_tau_2", "activation": "tf.keras.activations.linear"},
    ],
}


@pytest.fixture
def setUp():
    tf.keras.backend.clear_session()
    return Policy_Network(**network_config, config_env_specifics=env_config)


def test_prediction(setUp):
    # Generate a batch of dummy input data
    dummy_inputs = np.random.rand(
        5, len(env_config["state_variables"])
    )  # Batch size of 5, matching input_space
    # Use the setUp fixture to get an instance of your Policy_Network
    model = setUp

    # Perform prediction using the model
    predictions = model.predict(dummy_inputs)

    # Check if the output predictions have the correct shape
    assert predictions.shape == (
        5,
        len(env_config["action_variables"]),
    ), "The shape of the output predictions is incorrect."
