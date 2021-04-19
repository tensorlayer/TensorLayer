#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = ['Input', '_InputLayer']


class _InputLayer(Module):
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

    def __init__(self, shape, dtype=tl.float32, name=None):
        super(_InputLayer, self).__init__(name)

        # if isinstance(dtype, str):
        #     try:
        #         dtype = eval(dtype)
        #     except Exception as e:
        #         raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))
        # if not isinstance(dtype, tl.DType):
        #     raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))

        logging.info("Input  %s: %s" % (self.name, str(shape)))
        self.shape = shape  # shape is needed in __repr__
        self.dtype = dtype
        self.shape_without_none = [_ if _ is not None else 1 for _ in shape]
        self.outputs = tl.initializers.ones()(self.shape_without_none, dtype=self.dtype)
        self._built = True
        # self._add_node(outputs, outputs)

    def __repr__(self):
        s = 'Input(shape=%s' % str(self.shape)
        if self.name is not None:
            s += (', name=\'%s\'' % self.name)
        s += ')'
        return s

    def __call__(self, *args, **kwargs):
        # return super(_InputLayer, self).__call__(inputs)
        return self.outputs

    def build(self, inputs_shape):
        pass

    def forward(self):
        # tl.initializers.random_uniform()
        # tl.initializers.random_normal()
        # tl.initializers.truncated_normal()
        # tl.initializers.constant(2.0)
        # tl.initializers.He_Normal()
        # tl.initializers.He_Normal()
        # tl.initializers.zeros()
        # tl.initializers.ones()

        # outputs = self.inputs(self.shape_without_none, dtype=self.dtype)
        return self.outputs


def Input(shape, init=tl.initializers.ones(), dtype=tl.float32, name=None):
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
    outputs = input_layer(init)
    # outputs = input_layer._nodes[0].out_tensors[0]
    return outputs
