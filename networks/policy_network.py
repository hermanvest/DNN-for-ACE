import tensorflow as tf
from tensorflow.keras.layers import Dense

from typing import Any, Dict


class Policy_Network(tf.keras.Model):
    """
    __summary

    Parameters:
    - hidden_nodes (int): The number of nodes in each hidden layer.
    - hidden_activation_function (str): The activation function name for the hidden layers (e.g., 'relu', 'tanh').
    - output_activation_function (str): The activation function name for the output layer.
    - kernel_initializer_config (Dict[str, Any]): Configuration for the kernel initializer of each layer, allowing for detailed specification of initialization parameters.
    - config_env (Dict[str, Any]): Environment configuration dictionary, which must include 'action_variables' and 'state_variables' keys with lists indicating the names of each.

    Attributes:
    - _config_action_variables: Stores the action variables from the environment configuration.
    - _hidden1: The first hidden layer of the network.
    - _hidden2: The second hidden layer of the network.
    - _output_layer: The output layer of the network, sized to match the number of action variables.

    The class inherits from `tf.keras.Model`, making it compatible with TensorFlow's model training and evaluation utilities.
    """

    def __init__(
        self,
        hidden_nodes: int,
        hidden_activation_function: str,
        output_activation_function: str,
        kernel_initializer_config: Dict[str, Any],
        config_env: Dict[str, Any],
    ) -> None:
        super(Policy_Network, self).__init__()

        self._config_action_variables = config_env["action_variables"]
        input_space = len(config_env["state_variables"])
        output_space = len(config_env["action_variables"])

        self._hidden1 = Dense(
            hidden_nodes,
            activation=hidden_activation_function,
            input_shape=(input_space,),
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                **kernel_initializer_config
            ),
        )
        self._hidden2 = Dense(
            hidden_nodes,
            activation=hidden_activation_function,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                **kernel_initializer_config
            ),
        )
        self._output_layer = Dense(
            output_space,
            activation=output_activation_function,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                **kernel_initializer_config
            ),
        )

    @staticmethod
    def safe_sigmoid(x, epsilon=1e-6):
        """
        Applies a modified sigmoid activation function to the input tensor `x` that ensures the output is strictly within the range (epsilon, 1-epsilon). This modification is designed to prevent the output from reaching the exact values of 0 and 1, thus maintaining numerical stability, especially for operations that are sensitive to these boundary values, such as logarithms.

        The function works by slightly compressing the sigmoid's output range using the `epsilon` parameter. The output is scaled and translated so that it lies within the specified bounds, ensuring differentiability and avoiding issues with infinite or undefined gradients.

        Parameters:
            x (tf.Tensor): The input tensor for which to compute the modified sigmoid activation.
            epsilon (float, optional): A small constant used to define the lower and upper bounds of the sigmoid's output, preventing it from reaching 0 or 1. Defaults to 1e-6.

        Returns:
            tf.Tensor: The output tensor, with each element transformed by the modified sigmoid function to lie strictly within (epsilon, 1-epsilon).

        Example:
            >>> x = tf.constant([-10.0, 0.0, 10.0])
            >>> activated_x = safe_sigmoid(x)
            >>> print(activated_x.numpy())
        """
        return epsilon + (1 - 2 * epsilon) * tf.sigmoid(x)

    def apply_actionspecific_activations(self, unprocessed_output) -> Any:
        # Tensorflow calls this function once for each bath
        # Therefore unprocessed output will always be shape (None, num_actions)
        processed_outputs = []

        # Iterate over the action configurations to apply specified activations
        for a_i, action_information in enumerate(self._config_action_variables):
            # Extract the activation function if specified, else use linear (no change)
            activation_name = action_information.get(
                "activation", "linear"
            )  # Default to linear

            if activation_name == "sigmoid":
                activation_func = self.safe_sigmoid
            else:
                activation_func = getattr(tf.keras.activations, activation_name)

            # Apply the activation function to the corresponding slice of output
            processed_output = activation_func(unprocessed_output[:, a_i : a_i + 1])

            # Collect the processed output
            processed_outputs.append(processed_output)

        # Reassemble the processed outputs into a single tensor
        return tf.concat(processed_outputs, axis=1)

    def call(self, inputs) -> Any:
        # Standard feed forward dense neural network
        x = self._hidden1(inputs)
        x = self._hidden2(x)
        x = self._output_layer(x)

        # Action specific activation functions
        x = self.apply_actionspecific_activations(x)
        return x
