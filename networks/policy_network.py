# Deep learning related libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Type related libraries
from typing import Any

# Constants
from config import Config


class Policy_Network(tf.keras.Model):
    def __init__(self) -> None:
        super(Policy_Network, self).__init__()

        # Define the layers in the network
        self.dense1 = Dense(64, activation="relu", input_shape=(Config.STATE_SPACE,))
        self.dense2 = Dense(64, activation="relu")
        self.output_layer = Dense(Config.ACTION_SPACE, activation="sigmoid")

    def call(self, inputs) -> Any:
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
