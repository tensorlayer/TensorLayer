#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Layer, LayerNode

# from tensorlayer.layers.core import LayersConfig

__all__ = ['Input', '_InputLayer']


class _InputLayer(Layer):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    dtype: dtype
        The type of input values. By default, tf.float32.
    name : None or str
        A unique layer name.

    """

    def __init__(self, shape, dtype=tf.float32, name=None):  #'input'):
        # super(InputLayer, self).__init__(prev_layer=inputs, name=name)
        super(_InputLayer, self).__init__(name)

        if isinstance(dtype, str):
            try:
                dtype = eval(dtype)
            except Exception as e:
                raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))
        if not isinstance(dtype, tf.DType):
            raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))

        logging.info("Input  %s: %s" % (self.name, str(shape)))
        self.shape = shape  # shape is needed in __repr__

        shape_without_none = [_ if _ is not None else 1 for _ in shape]
        # self.outputs = self.forward(tl.initializers.random_normal()(shape_without_none))
        outputs = self.forward(tl.initializers.ones()(shape_without_none, dtype=dtype))

        self._built = True

        self._add_node(outputs, outputs)

    def __repr__(self):
        s = 'Input(shape=%s' % str(self.shape)
        if self.name is not None:
            s += (', name=\'%s\'' % self.name)
        s += ')'
        return s

    def __call__(self, inputs, *args, **kwargs):
        return super(_InputLayer, self).__call__(inputs)

    def build(self, inputs_shape):
        pass

    def forward(self, inputs):
        return inputs


def Input(shape, dtype=tf.float32, name=None):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    name : None or str
        A unique layer name.

    """
    input_layer = _InputLayer(shape, dtype=dtype, name=name)
    outputs = input_layer._nodes[0].out_tensors[0]
    return outputs
