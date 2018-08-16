#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(
            self,
            prev_layer=None,
            fn=None,
            fn_args=None,
            act=None,
            name='lambda_layer',
    ):
        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        self.prev_layer = prev_layer
        self.fn = fn
        self.act = act
        self.name = name

        super(LambdaLayer, self).__init__(fn_args=fn_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("fn: %s" % self.fn.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)
    def compile(self, prev_layer, is_train=True):

        super(LambdaLayer, self).compile(prev_layer)

        with tf.variable_scope(self.name) as vs:
            self.outputs = self.fn(self.inputs, **self.fn_args)
            self.outputs = self._apply_activation(self.outputs)

            self._local_weights = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


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
        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        self.prev_layer = layers
        self.fn = fn
        self.act = act
        self.name = name

        super(ElementwiseLambdaLayer, self).__init__(fn_args=fn_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("fn: %s" % self.fn.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)
    def compile(self, prev_layer, is_train=True):

        super(ElementwiseLambdaLayer, self).compile(prev_layer)

        with tf.variable_scope(self.name) as vs:
            self.outputs = self.fn(*self.inputs, **self.fn_args)
            self.outputs = self._apply_activation(self.outputs)

            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)
