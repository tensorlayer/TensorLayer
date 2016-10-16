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

    References
    ------------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models, Maas et al. (2013) <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_
    """
    with tf.name_scope(name) as scope:
        # x = tf.nn.relu(x)
        # m_x = tf.nn.relu(-x)
        # x -= alpha * m_x
        x = tf.maximum(x, alpha * x)
    return x

#Shortcut
lrelu = leaky_relu


## Alternatively we can use tl.layers.PReluLayer()
def prelu(x, channel_shared=False, W_init=tf.constant_initializer(value=0.0), W_init_args={}, restore=True, name="PReLU"):
    """ Parametric Rectified Linear Unit.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    channel_shared : `bool`. Single weight is shared by all channels
    W_init: weights initializer, default zero constant.
        The initializer for initializing the alphas.
    restore : `bool`. Restore or not alphas
    name : A name for this activation op (optional).

    Returns
    -------
    A `Tensor` with the same type as `x`.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """
    print(' prelu: untested !!!')
    if channel_shared:
        w_shape = (1,)
    else:
        w_shape = int(x._shape[-1:])

    with tf.name_scope(name) as scope:
        W_init = initializations.get(weights_init)()
        alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=W_init, **W_init_args )
        x = tf.nn.relu(x) + tf.mul(alphas, (x - tf.abs(x))) * 0.5

    return x
