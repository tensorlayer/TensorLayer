#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'DenseLayer',
]


class DenseLayer(Layer):
    """The :class:`DenseLayer` class is a fully connected layer.

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
    name : a str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, 800, act=tf.nn.relu, name='relu')

    Without native TensorLayer APIs, you can do as follow.

    >>> weight_matrix = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, weight_matrix) + b)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`FlattenLayer`.

    """

    def __init__(
        self,
        n_units=100,
        act=None,
        W_init=tf.truncated_normal_initializer(stddev=0.1),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        name='dense',
    ):

        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.name = name

        super(DenseLayer, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_units: %d" % self.n_units)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self._temp_data['inputs'].get_shape().ndims != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")

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

            self._temp_data['outputs'] = tf.matmul(self._temp_data['inputs'], weight_matrix)

            if self.b_init is not None:
                try:
                    b = self._get_tf_variable(
                        name='b',
                        shape=self.n_units,
                        dtype=self._temp_data['inputs'].dtype,
                        trainable=self._temp_data['is_train'],
                        initializer=self.b_init,
                        **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = self._get_tf_variable(
                        name='b',
                        dtype=self._temp_data['inputs'].dtype,
                        trainable=self._temp_data['is_train'],
                        initializer=self.b_init,
                        **self.b_init_args
                    )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
