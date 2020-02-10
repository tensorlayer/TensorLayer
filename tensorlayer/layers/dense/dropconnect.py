#! /usr/bin/python
# -*- coding: utf-8 -*-

import numbers

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

__all__ = [
    'DropconnectDense',
]


class DropconnectDense(Layer):
    """
    The :class:`DropconnectDense` class is :class:`Dense` with DropConnect
    behaviour which randomly removes connections between this layer and the previous
    layer according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : weights initializer
        The initializer for the weight matrix.
    b_init : biases initializer
        The initializer for the bias vector.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.Input([None, 784], name='input')
    >>> net = tl.layers.DropconnectDense(keep=0.8,
    ...         n_units=800, act=tf.nn.relu, name='relu1')(net)
    >>> net = tl.layers.DropconnectDense(keep=0.5,
    ...         n_units=800, act=tf.nn.relu, name='relu2')(net)
    >>> net = tl.layers.DropconnectDense(keep=0.5,
    ...         n_units=10, name='output')(net)

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`__

    """

    def __init__(
            self,
            keep=0.5,
            n_units=100,
            act=None,
            W_init=tl.initializers.truncated_normal(stddev=0.05),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None,  # 'dropconnect',
    ):
        super().__init__(name, act=act)

        if isinstance(keep, numbers.Real) and not (keep > 0 and keep <= 1):
            raise ValueError("keep must be a scalar tensor or a float in the " "range (0, 1], got %g" % keep)

        self.keep = keep
        self.n_units = n_units
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info(
            "DropconnectDense %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        s += ', keep={keep}'
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2")

        if self.in_channels is None:
            self.in_channels = inputs_shape[1]

        n_in = inputs_shape[-1]
        self.W = self._get_weights("weights", shape=(n_in, self.n_units), init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_units), init=self.b_init)

    def forward(self, inputs):
        W_dropcon = tf.nn.dropout(self.W, 1 - (self.keep))
        outputs = tf.matmul(inputs, W_dropcon)
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
