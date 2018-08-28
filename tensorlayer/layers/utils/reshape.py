#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'flatten_reshape',
]


def flatten_reshape(variable, name='flatten'):
    """Reshapes a high-dimension vector input.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row x mask_col x n_mask]

    Parameters
    ----------
    variable : TensorFlow variable or tensor
        The variable or tensor to be flatten.
    name : str
        A unique layer name.

    Returns
    -------
    Tensor
        Flatten Tensor

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    >>> # Convolution Layer with 32 filters and a kernel size of 5
    >>> network = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    >>> # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    >>> network = tf.layers.max_pooling2d(network, 2, 2)
    >>> print(network.get_shape()[:].as_list())
    >>> [None, 62, 62, 32]
    >>> network = tl.layers.flatten_reshape(network)
    >>> print(network.get_shape()[:].as_list()[1:])
    >>> [None, 123008]
    """
    dim = 1

    for d in variable.get_shape()[1:].as_list():
        dim *= d

    return tf.reshape(variable, shape=[-1, dim], name=name)
