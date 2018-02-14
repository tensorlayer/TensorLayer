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


class FlattenLayer(Layer):
    """
    The :class:`FlattenLayer` class is layer which reshape high-dimension
    input to a vector. Then we can apply DenseLayer, RNNLayer, ConcatLayer and
    etc on the top of it.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.Conv2dLayer(net,
    ...                    act = tf.nn.relu,
    ...                    shape = [5, 5, 32, 64],
    ...                    strides=[1, 1, 1, 1],
    ...                    padding='SAME',
    ...                    name ='cnn_layer')
    >>> net = tl.layers.Pool2dLayer(net,
    ...                    ksize=[1, 2, 2, 1],
    ...                    strides=[1, 2, 2, 1],
    ...                    padding='SAME',
    ...                    pool = tf.nn.max_pool,
    ...                    name ='pool_layer',)
    >>> net = tl.layers.FlattenLayer(net, name='flatten_layer')
    """

    def __init__(
            self,
            layer=None,
            name='flatten_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = flatten_reshape(self.inputs, name=name)
        self.n_units = int(self.outputs.get_shape()[-1])
        print("  [TL] FlattenLayer %s: %d" % (self.name, self.n_units))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class ReshapeLayer(Layer):
    """
    The :class:`ReshapeLayer` class is layer which reshape the tensor.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    shape : a list
        The output shape.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - The core of this layer is ``tf.reshape``.
    - Use TensorFlow only :
    >>> x = tf.placeholder(tf.float32, shape=[None, 3])
    >>> y = tf.reshape(x, shape=[-1, 3, 3])
    >>> sess = tf.InteractiveSession()
    >>> print(sess.run(y, feed_dict={x:[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]}))
    ... [[[ 1.  1.  1.]
    ... [ 2.  2.  2.]
    ... [ 3.  3.  3.]]
    ... [[ 4.  4.  4.]
    ... [ 5.  5.  5.]
    ... [ 6.  6.  6.]]]
    """

    def __init__(
            self,
            layer=None,
            shape=[],
            name='reshape_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = tf.reshape(self.inputs, shape=shape, name=name)
        print("  [TL] ReshapeLayer %s: %s" % (self.name, self.outputs.get_shape()))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class TransposeLayer(Layer):
    """
    The :class:`TransposeLayer` class transpose the dimension of a teneor, see `tf.transpose() <https://www.tensorflow.org/api_docs/python/tf/transpose>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    perm: list, a permutation of the dimensions
        Similar with numpy.transpose.
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            perm=None,
            name='transpose',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        assert perm is not None

        print("  [TL] TransposeLayer  %s: perm:%s" % (self.name, perm))
        # with tf.variable_scope(name) as vs:
        self.outputs = tf.transpose(self.inputs, perm=perm, name=name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )
