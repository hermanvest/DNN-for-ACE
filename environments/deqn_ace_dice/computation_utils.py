import tensorflow as tf


def custom_sigmoid(x, upper_bound, epsilon=1e-6):
    """
    Applies a modified sigmoid activation function to input tensor `x` that scales the output to be between `epsilon` and `upper_bound` minus `epsilon`. This function is useful for scenarios where the activation output needs to be constrained within a specific range, avoiding exact 0 and ensuring differentiability.

    The modification involves scaling the traditional sigmoid output (which lies in the range (0, 1)) to ensure that the result is strictly greater than 0 and less than `upper_bound`, with a minimum value determined by `epsilon`.

    Parameters:
        x (tf.Tensor): The input tensor for which to compute the activation.
        upper_bound (float): The upper limit for the activation function's output, ensuring that the output is strictly less than this value.
        epsilon (float, optional): A small constant used to adjust the lower and upper limits of the sigmoid output to avoid the exact 0 and `upper_bound` values, ensuring numerical stability. Defaults to 1e-6.

    Returns:
        tf.Tensor: The scaled sigmoid activation of `x`, constrained to be within the range (`epsilon`, `upper_bound` - `epsilon`).

    Example:
        >>> x = tf.constant([0.0, -1.0, 1.0, 5.0])
        >>> upper_bound = 5.0
        >>> activated_x = custom_sigmoid(x, upper_bound)
        >>> print(activated_x)
    """
    return upper_bound * (epsilon + (1 - 2 * epsilon) * tf.sigmoid(x))


def logit(y):
    """
    Computes the logit (inverse of sigmoid) function using TensorFlow.

    Args:
        y (tf.Tensor): The input tensor, should contain values in the range (0, 1).

    Returns:
        tf.Tensor: The logit of the input tensor.
    """
    return tf.math.log(y / (1 - y))
