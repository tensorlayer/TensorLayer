#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# from tensorlayer.layers.core import LayersConfig

__all__ = [
    'Dense',
]


class Dense(Layer):
    """The :class:`Dense` class is a fully connected layer.

    Parameters
    ----------
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([100, 50], name='input')
    >>> dense = tl.layers.Dense(n_units=800, act=tf.nn.relu, in_channels=50, name='dense_1')
    >>> print(dense)
    Dense(n_units=800, relu, in_channels='50', name='dense_1')
    >>> tensor = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='dense_2')(net)
    >>> print(tensor)
    tf.Tensor([...], shape=(100, 800), dtype=float32)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
            self,
            n_units,
            act=None,
            W_init=tl.initializers.truncated_normal(stddev=0.1),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None,  # 'dense',
    ):

        super(Dense, self).__init__(name)

        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(self.in_channels)
            self._built = True

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
        if self.in_channels:
            shape = [self.in_channels, self.n_units]
        else:
            self.in_channels = inputs_shape[1]
            shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)

    # @tf.function
    def forward(self, inputs):
        z = tf.matmul(inputs, self.W)
        if self.b_init:
            z = tf.add(z, self.b)
        if self.act:
            z = self.act(z)
        return z
