# -*- coding: utf-8 -*-

from .core import *


class ExpandDimsLayer(Layer):
    """
    The :class:`ExpandDimsLayer` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#expand_dims>`__ .

    Parameters
    ----------
    layer : :class:`Layer`
        The previous layer.
    axis : int
        The dimension index at which to expand the shape of input.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
            axis,
            name='expand_dims',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        logging.info("ExpandDimsLayer  %s: axis:%d" % (self.name, axis))
        with tf.variable_scope(name):
            try:  # TF12 TF1.0
                self.outputs = tf.expand_dims(self.inputs, axis=axis)
            except Exception:  # TF11
                self.outputs = tf.expand_dims(self.inputs, dim=axis)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#tile>`__ .

    Parameters
    ----------
    layer : :class:`Layer`
        The previous layer.
    multiples: tensor
        Must be one of the following types: int32, int64.
        1-D Length must be the same as the number of dimensions in input.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer=None,
            multiples=None,
            name='tile',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        logging.info("TileLayer  %s: multiples:%s" % (self.name, multiples))
        with tf.variable_scope(name):
            self.outputs = tf.tile(self.inputs, multiples=multiples)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )
