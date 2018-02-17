# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class LambdaLayer(Layer):
    """
    The :class:`LambdaLayer` class is a layer which is able to use the provided function.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    fn : a function
        The function that applies to the outputs of previous layer.
    fn_args : a dictionary
        The arguments for the function (option).
    name : str
        A unique layer name.

    Examples
    ---------
    - Non-parametric case
    >>> x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = LambdaLayer(net, lambda x: 2*x, name='lambda')

    - Parametric case, merge other wrappers into TensorLayer
    >>> from keras.layers import *
    >>> from tensorlayer.layers import *
    >>> def keras_block(x):
    >>>     x = Dropout(0.8)(x)
    >>>     x = Dense(800, activation='relu')(x)
    >>>     x = Dropout(0.5)(x)
    >>>     x = Dense(800, activation='relu')(x)
    >>>     x = Dropout(0.5)(x)
    >>>     logits = Dense(10, activation='linear')(x)
    >>>     return logits
    >>> net = InputLayer(x, name='input')
    >>> net = LambdaLayer(net, fn=keras_block, name='keras')
    """

    def __init__(
            self,
            layer=None,
            fn=None,
            fn_args={},
            name='lambda_layer',
    ):
        Layer.__init__(self, name=name)
        assert layer is not None
        assert fn is not None
        self.inputs = layer.outputs
        logging.info("LambdaLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
