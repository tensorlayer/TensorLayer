#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import quantize

__all__ = [
    'Sign',
]


class Sign(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    name : a str
        A unique layer name.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            name=None  # 'sign',
    ):
        super().__init__(name)
        logging.info("Sign  %s" % self.name)

        self.build()
        self._built = True

    def build(self, inputs_shape=None):
        pass

    def __repr__(self):
        s = ('{classname}(')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        # with tf.variable_scope(name):
        ## self.outputs = tl.act.sign(self.inputs)
        # self.outputs = quantize(self.inputs)
        outputs = quantize(inputs)
        return outputs
