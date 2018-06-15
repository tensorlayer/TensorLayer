#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'LambdaLayer',
    'ElementwiseLambdaLayer',
]


class LambdaLayer(Layer):
    """A layer that takes a user-defined function using TensorFlow Lambda, for multiple inputs see :class:`ElementwiseLambdaLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    fn : function
        The function that applies to the outputs of previous layer.
    fn_args : dictionary or None
        The arguments for the function (option).
    name : str
        A unique layer name.

    Examples
    ---------
    Non-parametric case

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.LambdaLayer(net, lambda x: 2*x, name='lambda')

    Parametric case, merge other wrappers into TensorLayer

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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            fn,
            fn_args=None,
            name='lambda_layer',
    ):

        super(LambdaLayer, self).__init__(prev_layer=prev_layer, fn_args=fn_args, name=name)

        logging.info("LambdaLayer  %s" % self.name)

        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **self.fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


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

    >>> import tensorflow as tf
    >>> import tensorlayer as tl

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
