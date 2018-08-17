#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils.quantization import quantize

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'SignLayer',
]


class SignLayer(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        name='sign',
    ):

        self.prev_layer = prev_layer
        self.name = name

        super(SignLayer, self).__init__()

    def compile(self, prev_layer, is_train=True):

        super(SignLayer, self).compile(prev_layer)

        with tf.variable_scope(self.name):
            self.outputs = quantize(self.inputs)

        self._add_layers(self.outputs)
