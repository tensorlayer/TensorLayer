#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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

    def __init__(
        self,
        keep=0.5,
        n_units=100,
        act=None,
        W_init=tf.truncated_normal_initializer(stddev=0.1),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        name='dropconnect_layer',
    ):

        self.keep = keep
        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.name = name

        super(DropconnectDenseLayer, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("keep: %f" % self.keep)
        except AttributeError:
            pass

        try:
            additional_str.append("n_units: %d" % self.n_units)
        except AttributeError:
            pass

        try:
            additional_str.append("dropconnect active: %r" % self._temp_data['is_train'])
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        if self._temp_data['inputs'].get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2")

        n_in = int(self._temp_data['inputs'].get_shape()[-1])

        with tf.variable_scope(self.name):
            weight_matrix = self._get_tf_variable(
                name='W',
                shape=(n_in, self.n_units),
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            if self._temp_data['is_train']:
                keep_plh = tf.placeholder(self._temp_data['inputs'].dtype, shape=())
                self._add_local_drop_plh(keep_plh, self.keep)

                LayersConfig.set_keep[self.name] = keep_plh

                weight_dropconnect = tf.nn.dropout(weight_matrix, keep_plh)

            else:
                weight_dropconnect = weight_matrix

            self._temp_data['outputs'] = tf.matmul(self._temp_data['inputs'], weight_dropconnect)

            if self.b_init:
                b = self._get_tf_variable(
                    name='b',
                    shape=(self.n_units, ),
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
