# -*- coding: utf-8 -*-

import tensorflow as tf

from .. import _logging as logging
from .core import *

from ..deprecation import deprecated_alias

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
    >>> x = tf.placeholder(tf.float32, (None, 100))
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.ExpandDimsLayer(n, 2)
    ... [None, 100, 1]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            axis,
            name='expand_dims',
    ):
        super(ExpandDimsLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("ExpandDimsLayer  %s: axis:%d" % (name, axis))

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name):
            try:  # TF12 TF1.0
                self.outputs = tf.expand_dims(self.inputs, axis=axis)
            except Exception:  # TF11
                self.outputs = tf.expand_dims(self.inputs, dim=axis)
        # self.all_layers = list(layer.all_layers)
        self.all_params = list(prev_layer.all_params)
        self.all_drop = dict(prev_layer.all_drop)
        self.all_layers.append(self.outputs)
        # self.all_params.extend( variables )


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
    >>> x = tf.placeholder(tf.float32, (None, 100))
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.ExpandDimsLayer(n, 2)
    >>> n = tl.layers.TileLayer(n, [-1, 1, 3])
    ... [None, 100, 3]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, multiples=None, name='tile'):
        super(TileLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("TileLayer  %s: multiples:%s" % (name, multiples))

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name):
            self.outputs = tf.tile(self.inputs, multiples=multiples)
        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        # self.all_params.extend( variables )
