# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class ExpandDimsLayer(Layer):
    """
    The :class:`ExpandDimsLayer` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#expand_dims>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    axis : int, 0-D (scalar).
        Specifies the dimension index at which to expand the shape of input.
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            axis=None,
            name='expand_dims',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  [TL] ExpandDimsLayer  %s: axis:%d" % (self.name, axis))
        with tf.variable_scope(name) as vs:
            try:  # TF12 TF1.0
                self.outputs = tf.expand_dims(self.inputs, axis=axis)
            except:  # TF11
                self.outputs = tf.expand_dims(self.inputs, dim=axis)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#tile>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    multiples: a list of int
        Must be one of the following types: int32, int64. 1-D. Length must be the same as the number of dimensions in input
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            multiples=None,
            name='tile',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  [TL] TileLayer  %s: multiples:%s" % (self.name, multiples))
        with tf.variable_scope(name) as vs:
            self.outputs = tf.tile(self.inputs, multiples=multiples)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )
