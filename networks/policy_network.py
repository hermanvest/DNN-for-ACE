# Deep learning related libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Type related libraries
from typing import Any, Dict


class Policy_Network(tf.keras.Model):
    def __init__(self, config: Dict[str, Any]) -> None:
        super(Policy_Network, self).__init__()

        self.dense1 = Dense(
            config["hidden_nodes"],
            activation="relu",
            input_shape=(config["input_space"],),
        )
        self.dense2 = Dense(config["hidden_nodes"], activation="relu")
        self.output_layer = Dense(config["output_space"], activation="sigmoid")

    def call(self, inputs) -> Any:
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
