#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorflow.python.training import moving_averages

from tensorlayer.layers.utils import quantize_active_overflow
from tensorlayer.layers.utils import quantize_weight_overflow

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'QuanDenseLayerWithBN',
]


class QuanDenseLayerWithBN(Layer):
    """The :class:`QuanDenseLayer` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and the output of the previous layer
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
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
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
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            W_init_args=None,
            name='quan_dense_with_bn',
    ):
        super(QuanDenseLayerWithBN, self).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, name=name)

        logging.info(
            "QuanDenseLayerWithBN  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        x = self.inputs
        self.inputs = quantize_active_overflow(self.inputs, bitA)
        self.n_units = n_units

        with tf.variable_scope(name):

            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            mid_out = tf.matmul(x, W)

            para_bn_shape = mid_out.get_shape()[-1:]

            if gamma_init:
                scale_para = tf.get_variable(
                    name='scale_para', shape=para_bn_shape, initializer=gamma_init, dtype=LayersConfig.tf_dtype,
                    trainable=is_train
                )
            else:
                scale_para = None

            if beta_init:
                offset_para = tf.get_variable(
                    name='offset_para', shape=para_bn_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype,
                    trainable=is_train
                )
            else:
                offset_para = None

            moving_mean = tf.get_variable(
                'moving_mean', para_bn_shape, initializer=tf.constant_initializer(1.), dtype=LayersConfig.tf_dtype,
                trainable=False
            )

            moving_variance = tf.get_variable(
                'moving_variance',
                para_bn_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
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

            w_fold = _w_fold(W, scale_para, var, epsilon)
            bias_fold = _bias_fold(offset_para, scale_para, mean, var, epsilon)

            W = quantize_weight_overflow(w_fold, bitW)
            # W = tl.act.sign(W)    # dont update ...

            # W = tf.Variable(W)

            self.outputs = tf.matmul(self.inputs, W)
            # self.outputs = xnor_gemm(self.inputs, W) # TODO

            self.outputs = tf.nn.bias_add(self.outputs, bias_fold, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        self._add_params([W, scale_para, offset_para, moving_mean, moving_variance])


def _w_fold(w, gama, var, epsilon):
    return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))


def _bias_fold(beta, gama, mean, var, epsilon):
    return tf.subtract(beta, tf.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))
