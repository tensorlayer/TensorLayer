#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import flatten_reshape

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'FlattenLayer',
    'ReshapeLayer',
    'TransposeLayer',
]


class FlattenLayer(Layer):
    """A layer that reshapes high-dimension input into a vector.

    Then we often apply DenseLayer, RNNLayer, ConcatLayer and etc on the top of a flatten layer.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.FlattenLayer(net, name='flatten')
    [?, 784]

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='flatten'):
        super(FlattenLayer, self).__init__(prev_layer=prev_layer, name=name)

        _out = flatten_reshape(self.inputs, name=name)
        self.n_units = int(_out.get_shape()[-1])

        logging.info("FlattenLayer %s: %d" % (self.name, self.n_units))

        self.outputs = _out

        self._add_layers(self.outputs)


class ReshapeLayer(Layer):
    """A layer that reshapes a given tensor.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    shape : tuple of int
        The output shape, see ``tf.reshape``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(None, 784))
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.ReshapeLayer(net, [-1, 28, 28, 1], name='reshape')
    >>> print(net.outputs)
    (?, 28, 28, 1)

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, shape, name='reshape'):
        super(ReshapeLayer, self).__init__(prev_layer=prev_layer, name=name)

        if not shape:
            raise ValueError("Shape list can not be empty")

        self.outputs = tf.reshape(self.inputs, shape=shape, name=name)
        self._add_layers(self.outputs)

        logging.info("ReshapeLayer %s: %s" % (self.name, self.outputs.get_shape()))


class TransposeLayer(Layer):
    """A layer that transposes the dimension of a tensor.

    See `tf.transpose() <https://www.tensorflow.org/api_docs/python/tf/transpose>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    perm: list of int
        The permutation of the dimensions, similar with ``numpy.transpose``.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.TransposeLayer(net, perm=[0, 1, 3, 2], name='trans')
    [None, 28, 1, 28]

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, perm, name='transpose'):

        if perm is None:
            raise AssertionError("The `perm` argument cannot be None")

        super(TransposeLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("TransposeLayer  %s: perm: %s" % (self.name, perm))

        self.outputs = tf.transpose(self.inputs, perm=perm, name=name)
        self._add_layers(self.outputs)
