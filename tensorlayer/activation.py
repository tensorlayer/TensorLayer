#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf

def identity(x, name=None):
    """The identity activation function

    Parameters
    ----------
    x : a tensor input
        input(s)


    Returns
    --------
    A `Tensor` with the same type as `x`.
    """
    return x

# Shortcut
linear = identity


def ramp(x=None, v_min=0, v_max=1, name=None):
    """The ramp activation function.

    Parameters
    ----------
    x : a tensor input
        input(s)
    v_min : float
        if input(s) smaller than v_min, change inputs to v_min
    v_max : float
        if input(s) greater than v_max, change inputs to v_max
    name : a string or None
        An optional name to attach to this activation function.


    Returns
    --------
    A `Tensor` with the same type as `x`.
    """
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)

def leaky_relu(x=None, alpha=0.1, name="LeakyReLU"):
    """The LeakyReLU.

    Modified version of ReLU, introducing a nonzero gradient for negative
    input.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    alpha : `float`. slope.
    name : a string or None
        An optional name to attach to this activation function.

    Returns
    --------
    A `Tensor` with the same type as `x`.

    References
    ------------
    - Rectifier Nonlinearities Improve Neural Network Acoustic Models,
        Maas et al. (2013).

    Links
    --------
    `Relu Hybrid ICML 2013 <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_

    """

    # If incoming Tensor has a scope, this op is defined inside it
    # i_scope = ""
    # if hasattr(x, 'scope'):
    #     if x.scope: i_scope = x.scope
    with tf.name_scope(name) as scope:
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x
    # x.scope = scope

    return x
