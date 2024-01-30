# Deep learning related libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Type related libraries
from typing import Any, Dict


class Policy_Network(tf.keras.Model):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(Policy_Network, self).__init__()

        self._dense1 = Dense(
            config["hidden_nodes"],
            activation="relu",
            input_shape=(config["input_space"],),
        )
        self._dense2 = Dense(config["hidden_nodes"], activation="relu")
        self._output_layer = Dense(config["output_space"], activation="sigmoid")

    def call(self, inputs) -> Any:
        x = self._dense1(inputs)
        x = self._dense2(x)
        return self._output_layer(x)
