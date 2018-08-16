#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils.reshape import flatten_reshape

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(self, prev_layer=None, name='flatten'):

        self.prev_layer = prev_layer
        self.name = name

        super(FlattenLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)
    def compile(self, prev_layer, is_train=True):

        _out = flatten_reshape(prev_layer.outputs, name=self.name)
        self.out_shape = _out.shape

        super(FlattenLayer, self).compile(prev_layer)

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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(self, prev_layer=None, shape=list(), name='reshape'):

        if not shape:
            raise ValueError("Shape list can not be empty")

        self.prev_layer = prev_layer
        self._shape = shape
        self.name = name

        super(ReshapeLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)
    def compile(self, prev_layer, is_train=True):

        _out = tf.reshape(prev_layer.outputs, shape=self._shape, name=self.name)
        self.out_shape = _out.shape

        super(ReshapeLayer, self).compile(prev_layer)

        self.outputs = _out
        self._add_layers(self.outputs)


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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(self, prev_layer=None, perm=None, name='transpose'):

        if perm is None:
            raise AssertionError("The `perm` argument cannot be None")

        self.prev_layer = prev_layer
        self.perm = perm
        self.name = name

        super(TransposeLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("perm: %s" % self.perm)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)
    def compile(self, prev_layer, is_train=True):

        _out = tf.transpose(prev_layer.outputs, perm=self.perm, name=self.name)
        self.out_shape = _out.shape

        super(TransposeLayer, self).compile(prev_layer)

        self.outputs = _out
        self._add_layers(self.outputs)
