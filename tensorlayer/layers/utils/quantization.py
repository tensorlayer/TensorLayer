#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'cabs', 'w_fold', 'bias_fold', 'quantize', 'quantize_active', 'quantize_dorefa', 'quantize_grad',
    'quantize_overflow', 'quantize_weight', 'quantize_active_overflow', 'quantize_weight_overflow'
]


def cabs(x):
    return tf.minimum(tf.abs(x), 1.0, name='cabs')


def bias_fold(beta, gama, mean, var, epsilon):
    return tf.subtract(beta, tf.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))


def w_fold(w, gama, var, epsilon):
    return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))


def quantize(x):
    # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
    with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
        return tf.sign(x)


def quantize_active(x, bitA):
    bypass = False
    bypass += (bitA == 64 and x.dtype == tf.float64)
    bypass += (bitA == 32 and x.dtype == tf.float32)
    bypass += (bitA == 16 and x.dtype == tf.float16)

    if bypass:
        return x

    return quantize_dorefa(x, bitA)


def quantize_dorefa(x, bitA):
    G = tf.get_default_graph()
    n = float(2**bitA - 1)

    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


@tf.RegisterGradient("TL_Sign_QuantizeGrad")
def quantize_grad(op, grad):
    """Clip and binarize tensor using the straight through estimator (STE) for the gradient."""
    return tf.clip_by_value(grad, -1, 1)


def quantize_overflow(x, k):
    G = tf.get_default_graph()
    n = float(2**k - 1)
    max_value = tf.reduce_max(x)
    min_value = tf.reduce_min(x)
    with G.gradient_override_map({"Round": "Identity"}):
        step = tf.stop_gradient((max_value - min_value) / n)
        return tf.round((tf.maximum(tf.minimum(x, max_value), min_value) - min_value) / step) * step + min_value


def quantize_weight(x, bitW, force_quantization=False):
    G = tf.get_default_graph()
    if bitW == 32 and not force_quantization:
        return x
    if bitW == 1:  # BWN
        with G.gradient_override_map({"Sign": "Identity"}):
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
            return tf.sign(x / E) * E
    x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)  # it seems as though most weights are within -1 to 1 region anyways
    return 2 * quantize_dorefa(x, bitW) - 1


def quantize_active_overflow(x, bitA):
    if bitA == 32:
        return x
    return quantize_overflow(x, bitA)


def quantize_weight_overflow(x, bitW):
    if bitW == 32:
        return x
    return quantize_overflow(x, bitW)
