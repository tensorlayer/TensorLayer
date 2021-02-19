#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

IN_F = 2
IN_JOINTS = 17
OUT_JOINTS = 17
neighbour_matrix = np.array(
    [
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0.],
        [1., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
        [1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.]
    ]
)


def mask_weight(weight):
    weights = tf.clip_by_norm(weight, 1)
    L = neighbour_matrix.T
    mask = tf.constant(L)
    input_size, output_size = weights.get_shape()
    input_size, output_size = int(input_size), int(output_size)
    assert input_size % IN_JOINTS == 0 and output_size % IN_JOINTS == 0
    in_F = int(input_size / IN_JOINTS)
    out_F = int(output_size / IN_JOINTS)
    weights = tf.reshape(weights, [IN_JOINTS, in_F, IN_JOINTS, out_F])
    mask = tf.reshape(mask, [IN_JOINTS, 1, IN_JOINTS, 1])

    weights = tf.cast(weights, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    masked_weights = weights * mask
    masked_weights = tf.reshape(masked_weights, [input_size, output_size])
    return masked_weights
