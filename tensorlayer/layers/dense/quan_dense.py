#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils.quantization import quantize_active_overflow
from tensorlayer.layers.utils.quantization import quantize_weight_overflow

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'QuantizedDense',
]


class QuantizedDense(Layer):
    """The :class:`QuantizedDense` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and
    the output of the previous layer are 'bitA' bits while inferencing.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    gemmlowp_at_inference : boolean
        If True, use gemmlowp instead of ``tf.matmul`` (gemm) for inference. (TODO).
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

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            bitW=8,
            bitA=8,
            gemmlowp_at_inference=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='quantized_dense',
    ):
        super(QuantizedDense, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "QuantizedDense  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.inputs = quantize_active_overflow(self.inputs, bitA)
        self.n_units = n_units

        with tf.variable_scope(name):

            W = self._get_tf_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=self.inputs.dtype, **self.W_init_args
            )

            W = quantize_weight_overflow(W, bitW)

            self.outputs = tf.matmul(self.inputs, W)

            if b_init is not None:
                try:
                    b = self._get_tf_variable(
                        name='b', shape=(n_units,), initializer=b_init, dtype=self.inputs.dtype, **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = self._get_tf_variable(name='b', initializer=b_init, dtype=self.inputs.dtype, **self.b_init_args)

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init is not None:
            self._add_params([W, b])
        else:
            self._add_params(W)
