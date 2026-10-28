import tensorflow as tf

from ace_deqn.model import ACTION_NAMES


def build_policy_network(config: dict) -> tf.keras.Model:
    """Feed-forward policy network: preprocessed state (7) -> raw actions (11).

    The raw outputs are mapped onto bounded controls and scaled costates by
    `AceDice.decode_actions`.
    """
    net = config["network"]
    initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform", seed=net["seed"]
    )
    inputs = tf.keras.Input(shape=(7,))
    h = inputs
    for _ in range(net["hidden_layers"]):
        h = tf.keras.layers.Dense(
            net["hidden_nodes"], activation=net["activation"], kernel_initializer=initializer
        )(h)
    outputs = tf.keras.layers.Dense(len(ACTION_NAMES), kernel_initializer=initializer)(h)
    return tf.keras.Model(inputs, outputs)
