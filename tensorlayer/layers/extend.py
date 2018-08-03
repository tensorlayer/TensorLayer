#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

__all__ = [
    'ExpandDimsLayer',
    'TileLayer',
]


class ExpandDimsLayer(Layer):
    """
    The :class:`ExpandDimsLayer` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/tf/expand_dims>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    axis : int
        The dimension index at which to expand the shape of input.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, (None, 100))
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.ExpandDimsLayer(n, 2)
    [None, 100, 1]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            axis=0,
            name='expand_dims',
    ):

        self.prev_layer = prev_layer
        self.axis = axis
        self.name = name

        super(ExpandDimsLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("axis: %d" % self.axis)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        self._parse_inputs(prev_layer)

        with tf.variable_scope(self.name):
            self.outputs = tf.expand_dims(self.inputs, axis=self.axis)
            self.out_shape = self.outputs.shape

        self._add_layers(self.outputs)

        super(ExpandDimsLayer, self).__call__(prev_layer)


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/tf/tile>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    multiples: tensor
        Must be one of the following types: int32, int64.
        1-D Length must be the same as the number of dimensions in input.
    name : str
        A unique layer name.


    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, (None, 100))
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.ExpandDimsLayer(n, 2)
    >>> n = tl.layers.TileLayer(n, [-1, 1, 3])
    [None, 100, 3]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer=None, multiples=None, name='tile'):

        self.prev_layer = prev_layer
        self.multiples = multiples
        self.name = name

        super(TileLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("multiples: %s" % self.multiples)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        self._parse_inputs(prev_layer)

        with tf.variable_scope(self.name):
            self.outputs = tf.tile(prev_layer.outputs, multiples=self.multiples)
            self.out_shape = self.outputs.shape

        self._add_layers(self.outputs)

        super(TileLayer, self).__call__(prev_layer)
