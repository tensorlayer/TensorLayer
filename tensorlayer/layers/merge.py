# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class ConcatLayer(Layer):
    """
    The :class:`ConcatLayer` class is layer which concat (merge) two or more tensor by given axis..

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    concat_dim : int
        Dimension along which to concatenate.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    ----------
    >>> sess = tf.InteractiveSession()
    >>> x = tf.placeholder(tf.float32, shape=[None, 784])
    >>> inputs = tl.layers.InputLayer(x, name='input_layer')
    >>> net1 = tl.layers.DenseLayer(inputs, 800, act=tf.nn.relu, name='relu1_1')
    >>> net2 = tl.layers.DenseLayer(inputs, 300, act=tf.nn.relu, name='relu2_1')
    >>> net = tl.layers.ConcatLayer([net1, net2], 1, name ='concat_layer')
    ...   InputLayer input_layer (?, 784)
    ...   DenseLayer relu1_1: 800, relu
    ...   DenseLayer relu2_1: 300, relu
    ...   ConcatLayer concat_layer, 1100
    >>> tl.layers.initialize_global_variables(sess)
    >>> net.print_params()
    ...     param 0: (784, 800) (mean: 0.000021, median: -0.000020 std: 0.035525)
    ...     param 1: (800,)     (mean: 0.000000, median: 0.000000  std: 0.000000)
    ...     param 2: (784, 300) (mean: 0.000000, median: -0.000048 std: 0.042947)
    ...     param 3: (300,)     (mean: 0.000000, median: 0.000000  std: 0.000000)
    ...     num of params: 863500
    >>> net.print_layers()
    ...     layer 0: ("Relu:0", shape=(?, 800), dtype=float32)
    ...     layer 1: Tensor("Relu_1:0", shape=(?, 300), dtype=float32)
    """

    def __init__(
            self,
            layer=[],
            concat_dim=1,
            name='concat_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)
        try:  # TF1.0
            self.outputs = tf.concat(self.inputs, concat_dim, name=name)
        except:  # TF0.12
            self.outputs = tf.concat(concat_dim, self.inputs, name=name)

        logging.info("ConcatLayer %s: axis: %d" % (self.name, concat_dim))

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        #self.all_drop = list_remove_repeat(self.all_drop) # it is a dict


class ElementwiseLayer(Layer):
    """
    The :class:`ElementwiseLayer` class combines multiple :class:`Layer` which have the same output shapes by a given elemwise-wise operation.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    combine_fn : a TensorFlow elemwise-merge function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`_ .
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - AND Logic
    >>> net_0 = tl.layers.DenseLayer(net_0, n_units=500,
    ...                        act = tf.nn.relu, name='net_0')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=500,
    ...                        act = tf.nn.relu, name='net_1')
    >>> net_com = tl.layers.ElementwiseLayer(layer = [net_0, net_1],
    ...                         combine_fn = tf.minimum,
    ...                         name = 'combine_layer')
    """

    def __init__(
            self,
            layer=[],
            combine_fn=tf.minimum,
            name='elementwise_layer',
    ):
        Layer.__init__(self, name=name)

        logging.info("ElementwiseLayer %s: size:%s fn:%s" % (self.name, layer[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layer[0].outputs
        # logging.info(self.outputs._shape, type(self.outputs._shape))
        for l in layer[1:]:
            assert str(self.outputs.get_shape()) == str(
                l.outputs.get_shape()), "Hint: the input shapes should be the same. %s != %s" % (self.outputs.get_shape(), str(l.outputs.get_shape()))
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)

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
