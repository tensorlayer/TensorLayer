#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


def identity(x, name=None):
    """The identity activation function, Shortcut is ``linear``.

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


def leaky_relu(x=None, alpha=0.1, name="lrelu"):
    """The LeakyReLU, Shortcut is ``lrelu``.

    Modified version of ReLU, introducing a nonzero gradient for negative
    input.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    alpha : `float`. slope.
    name : a string or None
        An optional name to attach to this activation function.

    Examples
    ---------
    >>> network = tl.layers.DenseLayer(network, n_units=100, name = 'dense_lrelu',
    ...                 act= lambda x : tl.act.lrelu(x, 0.2))

    References
    ------------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models, Maas et al. (2013) <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`_
    """
    # with tf.name_scope(name) as scope:
    # x = tf.nn.relu(x)
    # m_x = tf.nn.relu(-x)
    # x -= alpha * m_x
    x = tf.maximum(x, alpha * x, name=name)
    return x


#Shortcut
lrelu = leaky_relu


def swish(x, name='swish'):
    """The Swish function, see `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`_.

    Parameters
    ----------
    x : a tensor input
        input(s)

    Returns
    --------
    A `Tensor` with the same type as `x`.
    """
    with tf.name_scope(name) as scope:
        x = tf.nn.sigmoid(x) * x
    return x


def pixel_wise_softmax(output, name='pixel_wise_softmax'):
    """Return the softmax outputs of images, every pixels have multiple label, the sum of a pixel is 1.
    Usually be used for image segmentation.

    Parameters
    ------------
    output : tensor
        - For 2d image, 4D tensor [batch_size, height, weight, channel], channel >= 2.
        - For 3d image, 5D tensor [batch_size, depth, height, weight, channel], channel >= 2.

    Examples
    ---------
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - dice_coe(outputs, y_, epsilon=1e-5)

    References
    -----------
    - `tf.reverse <https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#reverse>`_
    """
    with tf.name_scope(name) as scope:
        return tf.nn.softmax(output)
        ## old implementation
        # exp_map = tf.exp(output)
        # if output.get_shape().ndims == 4:   # 2d image
        #     evidence = tf.add(exp_map, tf.reverse(exp_map, [False, False, False, True]))
        # elif output.get_shape().ndims == 5: # 3d image
        #     evidence = tf.add(exp_map, tf.reverse(exp_map, [False, False, False, False, True]))
        # else:
        #     raise Exception("output parameters should be 2d or 3d image, not %s" % str(output._shape))
        # return tf.div(exp_map, evidence)
