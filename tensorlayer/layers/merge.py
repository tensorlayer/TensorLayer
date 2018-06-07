#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import tl_logging as logging

__all__ = [
    'ConcatLayer',
    'ElementwiseLayer',
    'ElementwiseLambdaLayer',
]


class ConcatLayer(Layer):
    """A layer that concats multiple tensors according to given axis.

    Parameters
    ----------
    layers : list of :class:`Layer`
        List of layers to concatenate.
    concat_dim : int
        The dimension to concatenate.
    name : str
        A unique layer name.

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
    ... [TL]   param   0: relu1_1/W:0          (784, 800)         float32_ref
    ... [TL]   param   1: relu1_1/b:0          (800,)             float32_ref
    ... [TL]   param   2: relu2_1/W:0          (784, 300)         float32_ref
    ... [TL]   param   3: relu2_1/b:0          (300,)             float32_ref
    ...     num of params: 863500
    >>> net.print_layers()
    ... [TL]   layer   0: relu1_1/Relu:0       (?, 800)           float32
    ... [TL]   layer   1: relu2_1/Relu:0       (?, 300)           float32
    ... [TL]   layer   2: concat_layer:0       (?, 1100)          float32

    """

    def __init__(
            self,
            layers,
            concat_dim=-1,
            name='concat_layer',
    ):

        super(ConcatLayer, self).__init__(prev_layer=layers, name=name)

        logging.info("ConcatLayer %s: axis: %d" % (self.name, concat_dim))

        self.outputs = tf.concat(self.inputs, concat_dim, name=name)

        self._add_layers(self.outputs)


class ElementwiseLayer(Layer):
    """A layer that combines multiple :class:`Layer` that have the same output shapes
    according to an element-wise operation.

    Parameters
    ----------
    layers : list of :class:`Layer`
        The list of layers to combine.
    combine_fn : a TensorFlow element-wise combine function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`__ .
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net_0 = tl.layers.DenseLayer(inputs, n_units=500, act=tf.nn.relu, name='net_0')
    >>> net_1 = tl.layers.DenseLayer(inputs, n_units=500, act=tf.nn.relu, name='net_1')
    >>> net = tl.layers.ElementwiseLayer([net_0, net_1], combine_fn=tf.minimum, name='minimum')
    >>> net.print_params(False)
    ... [TL]   param   0: net_0/W:0            (784, 500)         float32_ref
    ... [TL]   param   1: net_0/b:0            (500,)             float32_ref
    ... [TL]   param   2: net_1/W:0            (784, 500)         float32_ref
    ... [TL]   param   3: net_1/b:0            (500,)             float32_ref
    >>> net.print_layers()
    ... [TL]   layer   0: net_0/Relu:0         (?, 500)           float32
    ... [TL]   layer   1: net_1/Relu:0         (?, 500)           float32
    ... [TL]   layer   2: minimum:0            (?, 500)           float32
    """

    def __init__(
            self,
            layers,
            combine_fn=tf.minimum,
            act=None,
            name='elementwise_layer',
    ):

        super(ElementwiseLayer, self).__init__(prev_layer=layers, name=name)
        logging.info(
            "ElementwiseLayer %s: size: %s fn: %s" % (self.name, layers[0].outputs.get_shape(), combine_fn.__name__)
        )

        self.outputs = layers[0].outputs

        for l in layers[1:]:
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)

        self.outputs = self._apply_activation(self.outputs)

        # for i in range(1, len(layers)):
        #     self._add_layers(list(layers[i].all_layers))
        #     self._add_params(list(layers[i].all_params))
        #     self.all_drop.update(dict(layers[i].all_drop))

        self._add_layers(self.outputs)


class ElementwiseLambdaLayer(Layer):
    """A layer that use a custom function to combine multiple :class:`Layer` inputs.

    Parameters
    ----------
    layers : list of :class:`Layer`
        The list of layers to combine.
    fn : function
        The function that applies to the outputs of previous layer.
    fn_args : dictionary or None
        The arguments for the function (option).
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    --------
    z = mean + noise * tf.exp(std * 0.5)

    >>> def func(noise, mean, std):
    >>>     return mean + noise * tf.exp(std * 0.5)
    >>> x = tf.placeholder(tf.float32, [None, 200])
    >>> noise_tensor = tf.random_normal(tf.stack([tf.shape(x)[0], 200]))
    >>> noise = tl.layers.InputLayer(noise_tensor)
    >>> net = tl.layers.InputLayer(x)
    >>> net = tl.layers.DenseLayer(net, n_units=200, act=tf.nn.relu, name='dense1')
    >>> mean = tl.layers.DenseLayer(net, n_units=200, name='mean')
    >>> std = tl.layers.DenseLayer(net, n_units=200, name='std')
    >>> z = tl.layers.ElementwiseLambdaLayer([noise, mean, std], fn=func, name='z')
    """

    def __init__(
            self,
            layers,
            fn,
            fn_args=None,
            act=None,
            name='elementwiselambda_layer',
    ):

        super(ElementwiseLambdaLayer, self).__init__(prev_layer=layers, act=act, fn_args=fn_args, name=name)
        logging.info("ElementwiseLambdaLayer %s" % self.name)

        with tf.variable_scope(name) as vs:
            self.outputs = self._apply_activation(fn(*self.inputs, **self.fn_args))

            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)
