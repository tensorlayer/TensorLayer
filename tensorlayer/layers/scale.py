#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Scale',
]


class Scale(Layer):
    """The :class:`Scale` class is for multipling a trainble scale value to the layer outputs. Usually be used on the output of binary net.

    Parameters
    ----------
    init_scale : float
        The initial value for the scale factor.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            init_scale=0.05,
            name='scale',
    ):
        # super(Scale, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.init_scale = init_scale
        logging.info("Scale  %s: init_scale: %f" % (self.name, self.init_scale))

    def build(self, input_shape):
        self.scale = self._get_weights(
            "scale", shape=[1], init=tf.compat.v1.initializers.constant(value=self.init_scale)
        )  #, init_args=self.W_init_args)
        # self.scale = tf.compat.v1.get_variable("scale", shape=[1], initializer=tf.compat.v1.initializers.constant(value=self.init_scale))
        # self.add_weights(self.scale)

    def forward(self, inputs):
        outputs = inputs * self.scale
        return outputs
