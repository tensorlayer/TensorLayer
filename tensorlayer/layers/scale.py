#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'ScaleLayer',
]


class ScaleLayer(Layer):
    """The :class:`AddScaleLayer` class is for multipling a trainble scale value to the layer outputs. Usually be used on the output of binary net.

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

    @auto_parse_inputs
    def compile(self, prev_layer):

        with tf.variable_scope(self.name):
            scale = self._get_tf_variable(
                name="scale",
                shape=[1],
                initializer=tf.constant_initializer(value=self.init_scale),
                dtype=self._temp_data['inputs'].dtype
            )
            self._temp_data['outputs'] = tf.multiply(self._temp_data['inputs'], scale)
