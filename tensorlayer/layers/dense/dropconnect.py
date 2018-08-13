#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'DropconnectDenseLayer',
]


class DropconnectDenseLayer(Layer):
    """
    The :class:`DropconnectDenseLayer` class is :class:`DenseLayer` with DropConnect
    behaviour which randomly removes connections between this layer and the previous
    layer according to a keeping probability.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.8,
    ...         n_units=800, act=tf.nn.relu, name='relu1')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.5,
    ...         n_units=800, act=tf.nn.relu, name='relu2')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.5,
    ...         n_units=10, name='output')

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            keep=0.5,
            n_units=100,
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dropconnect_layer',
    ):
        super(DropconnectDenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DropconnectDenseLayer %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )
            b = tf.get_variable(
                name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
            )
            # self.outputs = tf.matmul(self.inputs, W) + b

            LayersConfig.set_keep[name] = tf.placeholder(tf.float32)

            W_dropcon = tf.nn.dropout(W, LayersConfig.set_keep[name])

            self.outputs = self._apply_activation(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_drop.update({LayersConfig.set_keep[name]: keep})
        self._add_layers(self.outputs)
        self._add_params([W, b])
