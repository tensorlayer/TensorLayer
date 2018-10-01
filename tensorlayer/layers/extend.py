#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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

    def __init__(
        self,
        axis=0,
        name='expand_dims',
    ):

        self.axis = axis
        self.name = name

        super(ExpandDimsLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("axis: %d" % self.axis)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            self._temp_data['outputs'] = tf.expand_dims(self._temp_data['inputs'], axis=self.axis)


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/tf/tile>`__ .

    Parameters
    ----------
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

    def __init__(self, multiples=None, name='tile'):

        self.multiples = multiples
        self.name = name

        super(TileLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("multiples: %s" % self.multiples)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            self._temp_data['outputs'] = tf.tile(self._temp_data['inputs'], multiples=self.multiples)
