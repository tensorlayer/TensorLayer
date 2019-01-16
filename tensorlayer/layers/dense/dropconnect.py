#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.Input(x, name='input')
    >>> net = tl.layers.DropconnectDense(net, keep=0.8,
    ...         n_units=800, act=tf.nn.relu, name='relu1')
    >>> net = tl.layers.DropconnectDense(net, keep=0.5,
    ...         n_units=800, act=tf.nn.relu, name='relu2')
    >>> net = tl.layers.DropconnectDense(net, keep=0.5,
    ...         n_units=10, name='output')

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`__

    """

    def __init__(
            self,
            keep=0.5,
            n_units=100,
            act=None,
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.1),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None,  # 'dropconnect',
    ):
        # super(DropconnectDense, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.keep = keep
        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        logging.info(
            "DropconnectDense %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):

        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2")

        self.n_in = inputs_shape[-1]

        self._add_weight(scope_name=self.name, var_name="weights", shape=(n_in, self.n_units), init=self.W_init, init_args=self.W_init_args)
        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\W', shape=(self.n_in, self.n_units), initializer=self.W_init,
        #     dtype=LayersConfig.tf_dtype, **self.W_init_args
        # )
        if self.b_init:
            self._add_weight(scope_name=self.name, var_name="biases", shape=(self.n_units), init=self.b_init, init_args=self.b_init_args)
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\b', shape=(self.n_units), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
        #         **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):
        W_dropcon = tf.nn.dropout(self.weights, 1 - (self.keep))
        outputs = tf.matmul(inputs, W_dropcon)
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.biases, name='bias_add')
        outputs = self.act(outputs)
        return outputs
