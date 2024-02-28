import tensorflow as tf

from typing import Tuple


class Rl_Ace_Dice:
    def __init__(self) -> None:
        pass

    def step(self) -> Tuple[tf.Tensor, tf.Tensor]:

        observation = tf.constant(1, dtype=tf.float32)
        reward = tf.constant(1, dtype=tf.float32)

        return observation, reward

    def reset(self) -> tf.Tensor:
        observation = tf.constant(1, dtype=tf.float32)

        return observation
