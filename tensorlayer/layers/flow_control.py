# -*- coding: utf-8 -*-

import copy
import inspect
import random
import time
import warnings

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class MultiplexerLayer(Layer):
    """
    The :class:`MultiplexerLayer` selects one of several input and forwards the selected input into the output,
    see `tutorial_mnist_multiplexer.py`.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.


    Attributes
    -----------------------
    sel : a placeholder
        Input an int [0, inf], which input is the output

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    >>> y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    >>> # define the network
    >>> net_in = tl.layers.InputLayer(x, name='input_layer')
    >>> net_in = tl.layers.DropoutLayer(net_in, keep=0.8, name='drop1')
    >>> # net 0
    >>> net_0 = tl.layers.DenseLayer(net_in, n_units=800,
    ...                                act = tf.nn.relu, name='net0/relu1')
    >>> net_0 = tl.layers.DropoutLayer(net_0, keep=0.5, name='net0/drop2')
    >>> net_0 = tl.layers.DenseLayer(net_0, n_units=800,
    ...                                act = tf.nn.relu, name='net0/relu2')
    >>> # net 1
    >>> net_1 = tl.layers.DenseLayer(net_in, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu1')
    >>> net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop2')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu2')
    >>> net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop3')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu3')
    >>> # multiplexer
    >>> net_mux = tl.layers.MultiplexerLayer(layer = [net_0, net_1], name='mux_layer')
    >>> network = tl.layers.ReshapeLayer(net_mux, shape=[-1, 800], name='reshape_layer') #
    >>> network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    >>> # output layer
    >>> network = tl.layers.DenseLayer(network, n_units=10,
    ...                                act = tf.identity, name='output_layer')

    References
    ------------
    - See ``tf.pack() for TF0.12 or tf.stack() for TF1.0`` and ``tf.gather()`` at `TensorFlow - Slicing and Joining <https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#slicing-and-joining>`_
    """

    def __init__(self, layer=[], name='mux_layer'):
        Layer.__init__(self, name=name)
        self.n_inputs = len(layer)

        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)
        try:  ## TF1.0
            all_inputs = tf.stack(self.inputs, name=name)  # pack means concat a list of tensor in a new dim  # 1.2
        except:
            all_inputs = tf.pack(self.inputs, name=name)  # pack means concat a list of tensor in a new dim  # 1.2

        print("  [TL] MultiplexerLayer %s: n_inputs:%d" % (self.name, self.n_inputs))

        self.sel = tf.placeholder(tf.int32)
        self.outputs = tf.gather(all_inputs, self.sel, name=name)  # [sel, :, : ...] # 1.2

        # print(self.outputs, vars(self.outputs))
        #         # tf.reshape(self.outputs, shape=)
        # exit()
        # the same with ConcatLayer
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        # self.all_drop = list_remove_repeat(self.all_drop)


# class DemultiplexerLayer(Layer):
#     """
#     The :class:`DemultiplexerLayer` takes a single input and select one of many output lines, which is connected to the input.
#
#     Parameters
#     ----------
#     layer : a list of :class:`Layer` instances
#         The `Layer` class feeding into this layer.
#     n_outputs : an int
#         The number of output
#     name : a string or None
#         An optional name to attach to this layer.
#
#     Field (Class Variables)
#     -----------------------
#     sel : a placeholder
#         Input int [0, inf], the
#     outputs : a list of Tensor
#         A list of outputs
#
#     Examples
#     --------
#     >>>
#     """
#     def __init__(self,
#            layer = None,
#            name='demux_layer'):
#         Layer.__init__(self, name=name)
#         self.outputs = []
