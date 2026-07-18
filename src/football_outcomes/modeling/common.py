from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Lambda


def zero_vec_from_scalar_input(
    input_tensor,
    width: int,
    name: str,
):
    return Lambda(
        lambda tensor: tf.zeros(
            (
                tf.shape(tensor)[0],
                width,
            ),
            dtype=tf.float32,
        ),
        name=name,
    )(input_tensor)


def zero_mask_like(
    values,
    name: str,
):
    return Lambda(
        lambda tensor: tf.zeros_like(tensor),
        name=name,
    )(values)
