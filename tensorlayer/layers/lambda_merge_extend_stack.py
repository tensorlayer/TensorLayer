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

## Lambda
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


## Merge layer
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
    ...     [TL] InputLayer input_layer (?, 784)
    ...     [TL] DenseLayer relu1_1: 800, relu
    ...     [TL] DenseLayer relu2_1: 300, relu
    ...     [TL] ConcatLayer concat_layer, 1100
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

        print("  [TL] ConcatLayer %s: axis: %d" % (self.name, concat_dim))

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

        print("  [TL] ElementwiseLayer %s: size:%s fn:%s" % (self.name, layer[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layer[0].outputs
        # print(self.outputs._shape, type(self.outputs._shape))
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


## Extend
class ExpandDimsLayer(Layer):
    """
    The :class:`ExpandDimsLayer` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#expand_dims>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    axis : int, 0-D (scalar).
        Specifies the dimension index at which to expand the shape of input.
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            axis=None,
            name='expand_dims',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  [TL] ExpandDimsLayer  %s: axis:%d" % (self.name, axis))
        with tf.variable_scope(name) as vs:
            try:  # TF12 TF1.0
                self.outputs = tf.expand_dims(self.inputs, axis=axis)
            except:  # TF11
                self.outputs = tf.expand_dims(self.inputs, dim=axis)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#tile>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    multiples: a list of int
        Must be one of the following types: int32, int64. 1-D. Length must be the same as the number of dimensions in input
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            multiples=None,
            name='tile',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  [TL] TileLayer  %s: multiples:%s" % (self.name, multiples))
        with tf.variable_scope(name) as vs:
            self.outputs = tf.tile(self.inputs, multiples=multiples)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        # self.all_params.extend( variables )


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
