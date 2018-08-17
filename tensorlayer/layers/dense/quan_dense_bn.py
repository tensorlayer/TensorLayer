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

__all__ = [
    'QuantizedDenseWithBN',
]


class QuantizedDenseWithBN(Layer):
    """The :class:`QuantizedDense` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    is_train : boolean
        Is being used for training or inference.
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
    is_train : boolean
        Is being used for training or inference.
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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            decay=0.9,
            epsilon=1e-5,
            is_train=False,
            bitW=8,
            bitA=8,
            gamma_init=tf.ones_initializer,
            beta_init=tf.zeros_initializer,
            gemmlowp_at_inference=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            W_init_args=None,
            name='quantized_dense_with_bn',
    ):
        super(QuantizedDenseWithBN, self).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, name=name)

        logging.info(
            "QuantizedDenseWithBN  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        x = self.inputs
        self.inputs = quantize_active_overflow(self.inputs, bitA)
        self.n_units = n_units

        with tf.variable_scope(name):

            weight_matrix = self._get_tf_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=self.inputs.dtype, **self.W_init_args
            )

            mid_out = tf.matmul(x, weight_matrix)

            para_bn_shape = mid_out.get_shape()[-1:]

            if gamma_init:
                scale_para = self._get_tf_variable(
                    name='scale_para',
                    shape=para_bn_shape,
                    initializer=gamma_init,
                    dtype=self.inputs.dtype,
                    trainable=is_train
                )
            else:
                scale_para = None

            if beta_init:
                offset_para = self._get_tf_variable(
                    name='offset_para',
                    shape=para_bn_shape,
                    initializer=beta_init,
                    dtype=self.inputs.dtype,
                    trainable=is_train
                )
            else:
                offset_para = None

            moving_mean = self._get_tf_variable(
                'moving_mean',
                para_bn_shape,
                initializer=tf.constant_initializer(1.),
                dtype=self.inputs.dtype,
                trainable=False
            )

            moving_variance = self._get_tf_variable(
                'moving_variance',
                para_bn_shape,
                initializer=tf.constant_initializer(1.),
                dtype=self.inputs.dtype,
                trainable=False,
            )

            mean, variance = tf.nn.moments(mid_out, list(range(len(mid_out.get_shape()) - 1)))

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            _w_fold = w_fold(weight_matrix, scale_para, var, epsilon)
            _bias_fold = bias_fold(offset_para, scale_para, mean, var, epsilon)

            weight_matrix = quantize_weight_overflow(_w_fold, bitW)
            # weight_matrix = tl.act.sign(weight_matrix)    # dont update ...

            # weight_matrix = tf.Variable(weight_matrix)

            self.outputs = tf.matmul(self.inputs, weight_matrix)
            # self.outputs = xnor_gemm(self.inputs, weight_matrix) # TODO

            self.outputs = tf.nn.bias_add(self.outputs, _bias_fold, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        self._add_params([weight_matrix, scale_para, offset_para, moving_mean, moving_variance])
