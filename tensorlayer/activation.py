#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf


def identity(x):
    """The identity activation function.
    Shortcut is ``linear``.

    Parameters
    ----------
    x : Tensor
        input.

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    return x


def ramp(x, v_min=0, v_max=1, name=None):
    """The ramp activation function.

    Parameters
    ----------
    x : Tensor
        input.
    v_min : float
        cap input to v_min as a lower bound.
    v_max : float
        cap input to v_max as a upper bound.
    name : str
        The function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)


def leaky_relu(x, alpha=0.1, name="lrelu"):
    """The LeakyReLU, Shortcut is ``lrelu``.

    Modified version of ReLU, introducing a nonzero gradient for negative input.

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``,
        ``int16``, or ``int8``.
    alpha : float
        Slope.
    name : str
        The function name (optional).

    Examples
    --------
    >>> net = tl.layers.DenseLayer(net, 100, act=lambda x : tl.act.lrelu(x, 0.2), name='dense')

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ------------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models, Maas et al. (2013) <http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf>`__

    """
    # with tf.name_scope(name) as scope:
    # x = tf.nn.relu(x)
    # m_x = tf.nn.relu(-x)
    # x -= alpha * m_x
    x = tf.maximum(x, alpha * x, name=name)
    return x


def swish(x, name='swish'):
    """The Swish function.
     See `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`__.

    Parameters
    ----------
    x : Tensor
        input.
    name: str
        function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    with tf.name_scope(name):
        x = tf.nn.sigmoid(x) * x
    return x


def pixel_wise_softmax(x, name='pixel_wise_softmax'):
    """Return the softmax outputs of images, every pixels have multiple label, the sum of a pixel is 1.
    Usually be used for image segmentation.

    Parameters
    ----------
    x : Tensor
        input.
            - For 2d image, 4D tensor (batch_size, height, weight, channel), where channel >= 2.
            - For 3d image, 5D tensor (batch_size, depth, height, weight, channel), where channel >= 2.
    name : str
        function name (optional)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    Examples
    --------
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - dice_coe(outputs, y_, epsilon=1e-5)

    References
    ----------
    - `tf.reverse <https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#reverse>`__

    """
    with tf.name_scope(name):
        return tf.nn.softmax(x)


# Alias
linear = identity
lrelu = leaky_relu
