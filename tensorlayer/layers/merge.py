#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'ConcatLayer',
    'ElementwiseLayer',
]


class ConcatLayer(Layer):
    """A layer that concats multiple tensors according to given axis.

    Parameters
    ----------
    concat_dim : int
        The dimension to concatenate.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> sess = tf.InteractiveSession()
    >>> x = tf.placeholder(tf.float32, shape=[None, 784])
    >>> inputs = tl.layers.InputLayer(x, name='input_layer')
    [TL]   InputLayer input_layer (?, 784)
    >>> net1 = tl.layers.DenseLayer(inputs, 800, act=tf.nn.relu, name='relu1_1')
    [TL]   DenseLayer relu1_1: 800, relu
    >>> net2 = tl.layers.DenseLayer(inputs, 300, act=tf.nn.relu, name='relu2_1')
    [TL]   DenseLayer relu2_1: 300, relu
    >>> net = tl.layers.ConcatLayer([net1, net2], 1, name ='concat_layer')
    [TL]   ConcatLayer concat_layer, 1100
    >>> tl.layers.initialize_global_variables(sess)
    >>> net.print_weights()
    [TL]   param   0: relu1_1/W:0          (784, 800)         float32_ref
    [TL]   param   1: relu1_1/b:0          (800,)             float32_ref
    [TL]   param   2: relu2_1/W:0          (784, 300)         float32_ref
    [TL]   param   3: relu2_1/b:0          (300,)             float32_ref
        num of params: 863500
    >>> net.print_layers()
    [TL]   layer   0: relu1_1/Relu:0       (?, 800)           float32
    [TL]   layer   1: relu2_1/Relu:0       (?, 300)           float32
    [TL]   layer   2: concat_layer:0       (?, 1100)          float32

    """

    def __init__(
        self,
        concat_dim=-1,
        name='concat_layer',
    ):

        self.concat_dim = concat_dim
        self.name = name

        super(ConcatLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("axis: %s" % self.concat_dim)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = tf.concat(self._temp_data['inputs'], self.concat_dim, name=self.name)


class ElementwiseLayer(Layer):
    """A layer that combines multiple :class:`Layer` that have the same output shapes
    according to an element-wise operation.

    Parameters
    ----------
    combine_fn : a TensorFlow element-wise combine function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`__ .
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 784])
    >>> inputs = tl.layers.InputLayer(x, name='input_layer')
    >>> net_0 = tl.layers.DenseLayer(inputs, n_units=500, act=tf.nn.relu, name='net_0')
    >>> net_1 = tl.layers.DenseLayer(inputs, n_units=500, act=tf.nn.relu, name='net_1')
    >>> net = tl.layers.ElementwiseLayer([net_0, net_1], combine_fn=tf.minimum, name='minimum')
    >>> net.print_weights(False)
    [TL]   param   0: net_0/W:0            (784, 500)         float32_ref
    [TL]   param   1: net_0/b:0            (500,)             float32_ref
    [TL]   param   2: net_1/W:0            (784, 500)         float32_ref
    [TL]   param   3: net_1/b:0            (500,)             float32_ref
    >>> net.print_layers()
    [TL]   layer   0: net_0/Relu:0         (?, 500)           float32
    [TL]   layer   1: net_1/Relu:0         (?, 500)           float32
    [TL]   layer   2: minimum:0            (?, 500)           float32
    """

    def __init__(
        self,
        combine_fn=tf.minimum,
        act=None,
        name='elementwise_layer',
    ):

        self.combine_fn = combine_fn
        self.act = act
        self.name = name

        super(ElementwiseLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("fn: %s" % self.combine_fn.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = self._temp_data['inputs'][0]

        for layer in self._temp_data['inputs'][1:]:
            self._temp_data['outputs'] = self.combine_fn(self._temp_data['outputs'], layer, name=self.name)

        self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
