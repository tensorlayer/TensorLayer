#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'compute_alpha',
    'ternary_operation',
    '_compute_threshold',
]


def compute_alpha(x):
    """Computing the scale parameter."""
    threshold = _compute_threshold(x)
    alpha1_temp1 = tf.where(tf.greater(x, threshold), x, tf.zeros_like(x, x.dtype))
    alpha1_temp2 = tf.where(tf.less(x, -threshold), x, tf.zeros_like(x, x.dtype))
    alpha_array = tf.add(alpha1_temp1, alpha1_temp2, name=None)
    alpha_array_abs = tf.abs(alpha_array)

    alpha_array_abs1 = tf.where(
        tf.greater(alpha_array_abs, 0), tf.ones_like(alpha_array_abs, x.dtype), tf.zeros_like(alpha_array_abs, x.dtype)
    )

    alpha_sum = tf.reduce_sum(alpha_array_abs)
    n = tf.reduce_sum(alpha_array_abs1)

    return tf.div(alpha_sum, n)


def ternary_operation(x):
    """Ternary operation use threshold computed with weights."""
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):

        threshold = _compute_threshold(x)
        return tf.sign(tf.add(tf.sign(tf.add(x, threshold)), tf.sign(tf.add(x, -threshold))))


def _compute_threshold(x):
    """
    ref: https://github.com/XJTUWYD/TWN
    Computing the threshold.
    """
    x_sum = tf.reduce_sum(tf.abs(x), reduction_indices=None, keepdims=False, name=None)
    threshold = tf.div(x_sum, tf.cast(tf.size(x), x.dtype), name=None)

    return tf.multiply(threshold, 0.7, name=None)
