#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

__all__ = [
    'Input',
]


class Input(Layer):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    name : None or str
        A unique layer name.

    """

    def __init__(self, shape, name=None):  #'input'):
        # super(InputLayer, self).__init__(prev_layer=inputs, name=name)
        super(Input, self).__init__(name)

        logging.info("Input  %s: %s" % (self.name, str(shape)))

        shape_without_none = [_ if _ is not None else 1 for _ in shape]
        self.outputs = self.forward(tf.compat.v1.initializers.random_normal()(shape_without_none))

    def __call__(self, prev_layer):
        # FIXME: better exception raising
        raise Exception("__call__() of Input deactivated")

    def build(self, inputs_shape):
        # FIXME: documentation need double check
        """
        no weights to define
        """
        pass

    def forward(self, inputs):
        # FIXME: documentation need double check
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network.
        is_train: bool
            train (True) or test (False)
        """
        return inputs
