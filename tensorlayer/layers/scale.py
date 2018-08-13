#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'ScaleLayer',
]


class ScaleLayer(Layer):
    """The :class:`AddScaleLayer` class is for multipling a trainble scale value to the layer outputs. Usually be used on the output of binary net.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    init_scale : float
        The initial value for the scale factor.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            init_scale=0.05,
            name='scale',
    ):
        super(ScaleLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("ScaleLayer  %s: init_scale: %f" % (self.name, init_scale))

        with tf.variable_scope(name):
            # scale = tf.get_variable(name='scale_factor', init, trainable=True, )
            scale = tf.get_variable("scale", shape=[1], initializer=tf.constant_initializer(value=init_scale))
            self.outputs = self.inputs * scale

        self._add_layers(self.outputs)
        self._add_params(scale)
