#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(
        self,
        prev_layer=None,
        init_scale=0.05,
        name='scale',
    ):

        self.prev_layer = prev_layer
        self.init_scale = init_scale
        self.name = name

        super(ScaleLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("init_scale: %s" % self.init_scale)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self, prev_layer, is_train=True):

        super(ScaleLayer, self).compile(prev_layer)

        with tf.variable_scope(self.name):
            scale = self._get_tf_variable(
                "scale", shape=[1], initializer=tf.constant_initializer(value=self.init_scale)
            )
            self.outputs = tf.multiply(self.inputs, scale)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)
