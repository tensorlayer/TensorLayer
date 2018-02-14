#! /usr/bin/python
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


## Stack Unstack
class StackLayer(Layer):
    """
    The :class:`StackLayer` class is layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`_.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    axis : an int
        Dimension along which to concatenate.
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=[],
            axis=0,
            name='stack',
    ):
        Layer.__init__(self, name=name)
        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)

        self.outputs = tf.stack(self.inputs, axis=axis, name=name)

        print("  [TL] StackLayer %s: axis: %d" % (self.name, axis))

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


def UnStackLayer(
        layer=None,
        num=None,
        axis=0,
        name='unstack',
):
    """
    The :class:`UnStackLayer` is layer for unstacking the given dimension of a rank-R tensor into rank-(R-1) tensors., see `tf.unstack() <https://www.tensorflow.org/api_docs/python/tf/unstack>`_.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    num : an int
        The length of the dimension axis. Automatically inferred if None (the default).
    axis : an int
        Dimension along which to concatenate.
    name : a string or None
        An optional name to attach to this layer.

    Returns
    --------
    The list of layer objects unstacked from the input.
    """
    inputs = layer.outputs
    with tf.variable_scope(name) as vs:
        outputs = tf.unstack(inputs, num=num, axis=axis)

    print("  [TL] UnStackLayer %s: num: %s axis: %d, n_outputs: %d" % (name, num, axis, len(outputs)))

    net_new = []
    scope_name = tf.get_variable_scope().name
    if scope_name:
        whole_name = scope_name + '/' + name
    else:
        whole_name = name

    for i in range(len(outputs)):
        n = Layer(None, name=whole_name + str(i))
        n.outputs = outputs[i]
        n.all_layers = list(layer.all_layers)
        n.all_params = list(layer.all_params)
        n.all_drop = dict(layer.all_drop)
        n.all_layers.extend([inputs])

        net_new.append(n)

    return net_new
