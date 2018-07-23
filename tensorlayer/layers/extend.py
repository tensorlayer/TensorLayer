#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
            prev_layer,
            axis,
            name='expand_dims',
    ):
        super(ExpandDimsLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("ExpandDimsLayer  %s: axis: %d" % (self.name, axis))

        with tf.variable_scope(name):
            self.outputs = tf.expand_dims(self.inputs, axis=axis)

        self._add_layers(self.outputs)


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
    def __init__(self, prev_layer, multiples=None, name='tile'):

        super(TileLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("TileLayer  %s: multiples: %s" % (self.name, multiples))

        with tf.variable_scope(name):
            self.outputs = tf.tile(self.inputs, multiples=multiples)

        self._add_layers(self.outputs)
