import tensorflow as tf

from ace_uq.model import ACTION_NAMES, STATE_NAMES


def build_policy_network(config: dict) -> tf.keras.Model:
    """Feed-forward policy network: preprocessed state and parameters (10) -> raw actions (11)."""
    net = config["network"]
    initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform", seed=net["seed"]
    )
    inputs = tf.keras.Input(shape=(len(STATE_NAMES),))
    h = inputs
    for _ in range(net["hidden_layers"]):
        h = tf.keras.layers.Dense(
            net["hidden_nodes"], activation=net["activation"], kernel_initializer=initializer
        )(h)
    outputs = tf.keras.layers.Dense(len(ACTION_NAMES), kernel_initializer=initializer)(h)
    return tf.keras.Model(inputs, outputs)
