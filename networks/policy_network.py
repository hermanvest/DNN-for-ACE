# Deep learning related libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Type related libraries
from typing import Any


class Policy_Network(tf.keras.Model):
    def __init__(self) -> None:
        super(Policy_Network, self).__init__()
        self.state_size = 10
        self.action_size = 10

        # Define the layers in the network
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(64, activation="relu")
        self.output_layer = Dense(self.action_size, activation="sigmoid")

    def call(self, inputs) -> Any:
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
