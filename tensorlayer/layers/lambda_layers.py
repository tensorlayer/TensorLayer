#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'Lambda',
    'ElementwiseLambda',
]


class Lambda(Layer):
    """A layer that takes a user-defined function using TensorFlow Lambda, for multiple inputs see :class:`ElementwiseLambda`.

    Parameters
    ----------
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
    >>> net = tl.layers.Input(x, name='input')
    >>> net = tl.layers.Lambda(net, lambda x: 2*x, name='lambda')

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
    >>> net = Input(name='input')(x)
    >>> net = Lambda(fn=keras_block, name='keras')(net)

    """

    def __init__(
        self,
        fn,
        fn_args=None,
        act=None,
        name='lambda',
    ):
        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        self.fn = fn
        self.act = act
        self.name = name

        super(Lambda, self).__init__(fn_args=fn_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("fn: %s" % self.fn.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name) as vs:
            self._temp_data['outputs'] = self.fn(self._temp_data['inputs'], **self.fn_args)
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class ElementwiseLambda(Layer):
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
    >>> noise = tl.layers.Input()(noise_tensor)
    >>> net = tl.layers.Input()(x)
    >>> net = tl.layers.Dense(n_units=200, act=tf.nn.relu, name='dense1')(net)
    >>> mean = tl.layers.Dense(n_units=200, name='mean')(net)
    >>> std = tl.layers.Dense(n_units=200, name='std')(net)
    >>> z = tl.layers.ElementwiseLambda(fn=func, name='z')([noise, mean, std])
    """

    def __init__(
        self,
        fn,
        fn_args=None,
        act=None,
        name='elementwiselambda',
    ):
        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        self.fn = fn
        self.act = act
        self.name = name

        super(ElementwiseLambda, self).__init__(fn_args=fn_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("fn: %s" % self.fn.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            self._temp_data['outputs'] = self.fn(*self._temp_data['inputs'], **self.fn_args)
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
