#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Dense',
]


class Dense(Layer):
    # FIXME: documentation update needed
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
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    in_channels
    
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input(x, name='input')
    >>> net = tl.layers.Dense(net, 800, act=tf.nn.relu, name='relu')

    Without native TensorLayer APIs, you can do as follow.

    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
            self,
            n_units=100,
            act=None,
            # TODO: how to support more initializers
            W_init=tl.initializers.truncated_normal(stddev=0.1),
            b_init=tl.initializers.constant(value=0.0),
            # W_init=tf.compat.v1.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init=tf.compat.v1.initializers.truncated_normal,
            # b_init=tf.compat.v1.initializers.constant,
            # W_init_args={'stddev': 0.1},
            # b_init_args=None,
            in_channels=None,
            name=None,  # 'dense',
    ):

        # super(Dense, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)

        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(self.in_channels)
            self._built = True
        # self.W_init_args = W_init_args
        # self.b_init_args = b_init_args

        # self.n_in = int(self.inputs.get_shape()[-1])
        # self.inputs_shape = self.inputs.shape.as_list() #
        # self.outputs_shape = [self.inputs_shape[0], n_units]

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    '''
    def build(self, inputs):
        self.W = tf.get_variable(
            name='W', shape=(self.n_in, self.n_units), initializer=self.W_init, dtype=LayersConfig.tf_dtype,
            **self.W_init_args
        )
        if self.b_init is not None:
            try:
                self.b = tf.get_variable(
                    name='b', shape=(self.n_units), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )
            except Exception:  # If initializer is a constant, do not specify shape.
                self.b = tf.get_variable(
                    name='b', initializer=self.b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                )
        self.get_weights(self.W, self.b)
    '''

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
        if self.in_channels:
            shape = [self.in_channels, self.n_units]
        else:
            shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)
        # outputs_shape = [inputs_shape[0], self.n_units]
        # return outputs_shape

    '''
    def forward(self, inputs, is_train):
        outputs = tf.matmul(inputs, self.W)
        if self.b_init is not None:
            outputs = tf.add(z, self.b)
        outputs = self.act(outputs)
        return outputs
    '''

    def forward(self, inputs):
        z = tf.matmul(inputs, self.W)
        if self.b_init:
            z = tf.add(z, self.b)
        if self.act:
            z = self.act(z)
        return z
