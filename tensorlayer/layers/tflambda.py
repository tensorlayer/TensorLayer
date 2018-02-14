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
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    fn : a function
        The function that applies to the outputs of previous layer.
    fn_args : a dictionary
        The arguments for the function (option).
    name : a string or None
        An optional name to attach to this layer.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = LambdaLayer(net, lambda x: 2*x, name='lambda_layer')
    >>> y = net.outputs
    >>> sess = tf.InteractiveSession()
    >>> out = sess.run(y, feed_dict={x : [[1],[2]]})
    ... [[2],[4]]
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
        print("  [TL] LambdaLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
