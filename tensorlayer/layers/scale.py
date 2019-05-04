#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.initializers import constant
from tensorlayer.layers.core import Layer

__all__ = [
    'Scale',
]


class Scale(Layer):
    """The :class:`Scale` class is to multiple a trainable scale value to the layer outputs. Usually be used on the output of binary net.

    Parameters
    ----------
    init_scale : float
        The initial value for the scale factor.
    name : a str
        A unique layer name.

    Examples:
    ----------
    >>> inputs = tl.layers.Input([8, 3])
    >>> dense = tl.layers.Dense(n_units=10)(inputs)
    >>> outputs = tl.layers.Scale(init_scale=0.5)(dense)
    >>> model = tl.models.Model(inputs=inputs, outputs=[dense, outputs])
    >>> dense_out, scale_out = model(data, is_train=True)

    """

    def __init__(
            self,
            init_scale=0.05,
            name='scale',
    ):
        super(Scale, self).__init__(name)
        self.init_scale = init_scale

        self.build((None, ))
        self._built = True

        logging.info("Scale  %s: init_scale: %f" % (self.name, self.init_scale))

    def __repr__(self):
        s = '{classname}('
        s += 'init_scale={init_scale},'
        s += 'name={name}'
        s += ")"
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        self.scale = self._get_weights("scale", shape=[1], init=constant(value=self.init_scale))

    # @tf.function
    def forward(self, inputs):
        outputs = inputs * self.scale
        return outputs
