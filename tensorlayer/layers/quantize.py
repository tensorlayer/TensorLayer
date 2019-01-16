#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import quantize

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Sign',
]


class Sign(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    # prev_layer : :class:`Layer`
    #     Previous layer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            # prev_layer,
            name=None,  #'sign',
    ):
        # super(Sign, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        logging.info("Sign  %s" % self.name)

    def build(self, inputs_shape):
        pass

    def forward(inputs):
        # with tf.variable_scope(name):
        ## self.outputs = tl.act.sign(self.inputs)
        # self.outputs = quantize(self.inputs)
        outputs = quantize(inputs)
        return outputs
        # self._add_layers(self.outputs)
