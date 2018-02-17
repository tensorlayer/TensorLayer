# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class PadLayer(Layer):
    """
    The :class:`PadLayer` class is a Padding layer for any modes and dimensions.
    Please see `tf.pad <https://www.tensorflow.org/api_docs/python/tf/pad>`_ for usage.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    padding : a Tensor of type int32.
    mode : one of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    name : str
        A unique layer name.
    """

    def __init__(
            self,
            layer=None,
            paddings=None,
            mode='CONSTANT',
            name='pad_layer',
    ):
        Layer.__init__(self, name=name)
        assert paddings is not None, "paddings should be a Tensor of type int32. see https://www.tensorflow.org/api_docs/python/tf/pad"
        self.inputs = layer.outputs
        logging.info("PadLayer   %s: paddings:%s mode:%s" % (self.name, list(paddings), mode))

        self.outputs = tf.pad(self.inputs, paddings=paddings, mode=mode, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
