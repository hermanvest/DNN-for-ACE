import tensorflow as tf


def assert_valid(variable: tf.Tensor, explanation_of_variable: str):
    if variable is None:
        raise ValueError(f"Argument {explanation_of_variable} is None")

    if variable.dtype != tf.float32:
        raise TypeError(
            f"Invalid argument for {explanation_of_variable}. Expected tf.float32, got {variable.dtype}"
        )

    nan_mask = tf.math.is_nan(variable)
    is_scalar = tf.rank(variable) == 0
    if tf.reduce_any(nan_mask):
        if is_scalar:
            # Directly raise an error for scalar values
            raise ValueError(f"Nan encountered in {explanation_of_variable}.")
        else:
            # For tensors, proceed to find indices
            nan_indices = tf.where(nan_mask)
            raise ValueError(
                f"Nan encountered in {explanation_of_variable} at indices: {nan_indices.numpy().flatten()}"
            )
