#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorflow.python.training import moving_averages

from tensorlayer.layers.utils.quantization import bias_fold
from tensorlayer.layers.utils.quantization import w_fold
from tensorlayer.layers.utils.quantization import quantize_active_overflow
from tensorlayer.layers.utils.quantization import quantize_weight_overflow

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'QuantizedDenseWithBN',
]


class QuantizedDenseWithBN(Layer):
    """The :class:`QuantizedDense` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Parameters
    ----------
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
    gemmlowp_at_inference : boolean
        If True, use gemmlowp instead of ``tf.matmul`` (gemm) for inference. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    name : a str
        A unique layer name.

    """

    def __init__(
        self,
        n_units=100,
        act=None,
        decay=0.9,
        epsilon=1e-5,
        bitW=8,
        bitA=8,
        gemmlowp_at_inference=False,
        W_init=tf.truncated_normal_initializer(stddev=0.1),
        gamma_init=tf.ones_initializer,
        beta_init=tf.zeros_initializer,
        W_init_args=None,
        gamma_init_args=None,
        beta_init_args=None,
        name='quantized_dense_with_bn',
    ):

        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        self.n_units = n_units
        self.act = act
        self.decay = decay
        self.epsilon = epsilon
        self.bitW = bitW
        self.bitA = bitA
        self.gemmlowp_at_inference = gemmlowp_at_inference
        self.W_init = W_init
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        self.name = name

        super(QuantizedDenseWithBN, self).__init__(
            W_init_args=W_init_args, gamma_init_args=gamma_init_args, beta_init_args=beta_init_args
        )

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_units: %d" % self.n_units)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self._temp_data['inputs'].get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self._temp_data['inputs'].get_shape()[-1])

        quantized_inputs = quantize_active_overflow(self._temp_data['inputs'], self.bitA)

        self.n_units = self.n_units

        with tf.variable_scope(self.name):

            weight_matrix = self._get_tf_variable(
                name='W',
                shape=(n_in, self.n_units),
                dtype=quantized_inputs.dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            mid_out = tf.matmul(self._temp_data['inputs'], weight_matrix)

            para_bn_shape = mid_out.get_shape()[-1:]

            if self.gamma_init:
                scale_para = self._get_tf_variable(
                    name='scale_para',
                    shape=para_bn_shape,
                    dtype=quantized_inputs.dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.gamma_init,
                    **self.W_init_args
                )
            else:
                scale_para = None

            if self.beta_init:
                offset_para = self._get_tf_variable(
                    name='offset_para',
                    shape=para_bn_shape,
                    dtype=quantized_inputs.dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.beta_init,
                    **self.W_init_args
                )
            else:
                offset_para = None

            moving_mean = self._get_tf_variable(
                name='moving_mean',
                shape=para_bn_shape,
                dtype=quantized_inputs.dtype,
                trainable=False,
                initializer=tf.constant_initializer(1.)
            )

            moving_variance = self._get_tf_variable(
                name='moving_variance',
                shape=para_bn_shape,
                dtype=quantized_inputs.dtype,
                trainable=False,
                initializer=tf.constant_initializer(1.),
            )

            mean, variance = tf.nn.moments(mid_out, list(range(len(mid_out.get_shape()) - 1)))

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if self._temp_data['is_train']:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            _w_fold = w_fold(weight_matrix, scale_para, var, self.epsilon)
            _bias_fold = bias_fold(offset_para, scale_para, mean, var, self.epsilon)

            weight_matrix = quantize_weight_overflow(_w_fold, self.bitW)
            # weight_matrix = tl.act.sign(weight_matrix)    # dont update ...

            # weight_matrix = tf.Variable(weight_matrix)

            self._temp_data['outputs'] = tf.matmul(quantized_inputs, weight_matrix)
            # self._temp_data['outputs'] = xnor_gemm(quantized_inputs, weight_matrix) # TODO

            self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], _bias_fold, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
