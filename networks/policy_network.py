# Deep learning related libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Type related libraries
from typing import Any, Dict


class Policy_Network(tf.keras.Model):
    def __init__(
        self,
        hidden_nodes: int,
        hidden_activation_function: str,
        output_activation_function: str,
        kernel_initializer_config: Dict[str, Any],
        config_env_specifics: Dict[str, Any],
    ) -> None:
        super(Policy_Network, self).__init__()

        self._config_action_variables = config_env_specifics["action_variables"]
        input_space = len(config_env_specifics["state_variables"])
        output_space = len(config_env_specifics["action_variables"])

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

    def apply_actionspecific_activations(self, unprocessed_output) -> Any:
        # Output is in a Tensor shaped [batchsize, single_prediction_size]
        batch_size = unprocessed_output.shape[0]
        processed_output = unprocessed_output

        # Iterate through each policy configuration
        for a_i, action_information in enumerate(self._config_action_variables):
            # Check if an activation is specified
            if "activation" in action_information:
                activation_func = getattr(
                    tf.keras.activations,
                    action_information["activation"],
                    None,
                )

                # Apply the activation function if it's not None
                if activation_func:
                    # Apply activation function to each element for the current policy variable across the batch
                    activated_output = activation_func(processed_output[:, a_i])
                    # Update processed_output with the activated output for the current policy variable
                    indices = [[batch_i, a_i] for batch_i in range(batch_size)]

                    processed_output = tf.tensor_scatter_nd_update(
                        processed_output, indices, activated_output
                    )
        return processed_output

    def call(self, inputs) -> Any:
        # Standard feed forward dense neural network
        x = self._hidden1(inputs)
        x = self._hidden2(x)
        x = self._output_layer(x)

        # Action specific activation functions
        x = self.apply_actionspecific_activations(x)
        return x
