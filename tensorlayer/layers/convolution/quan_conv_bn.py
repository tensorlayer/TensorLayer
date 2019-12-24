#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.training import moving_averages

from tensorlayer import logging
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import (quantize_active_overflow, quantize_weight_overflow)

# from tensorlayer.layers.core import LayersConfig

__all__ = ['QuanConv2dWithBN']


class QuanConv2dWithBN(Layer):
    """The :class:`QuanConv2dWithBN` class is a quantized convolutional layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would keep the same.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
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
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.
    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> net = tl.layers.Input([None, 256, 256, 3])
    >>> net = tl.layers.QuanConv2dWithBN(net, 64, (5, 5), (1, 1),  act=tf.nn.relu, padding='SAME', name='qcnnbn1')(net)
    >>> print(net)
    >>> net = tl.layers.QuanConv2dWithBN(net, 64, (5, 5), (1, 1), padding='SAME', name='qcnnbn2')
    >>>
    ...
    """
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            act=None,
            decay=0.9,
            epsilon=1e-5,
            is_train=False,
            gamma_init=tl.initializers.ones,
            beta_init=tl.initializers.zeros,
            bitW=8,
            bitA=8,
            use_gemm=False,
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            W_init_args=None,
            data_format=None,
            name='quan_cnn2d_bn',
    ):
        super(QuanConv2dWithBN, self).__init__(act=act, name=name)
        self.prev_layer = prev_layer
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.decay = decay
        self.epsilon = epsilon
        self.is_train = is_train
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        self.bitW = bitW
        self.bitA = bitA
        self.use_gemm = use_gemm
        self.W_init = W_init
        self.W_init_args = W_init_args
        self.data_format = data_format
        logging.info(
            "QuanConv2dWithBN %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s " % (
                self.name, n_filter, filter_size, str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")

        try:
            pre_channel = int(prev_layer.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.warning("[warnings] unknow input channels, set to 1")


    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], 1]
            self._dilation_rate = [1, self._dilation_rate[0], self._dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.filter_size[0], self.filter_size[1], self.in_channels, self.n_filter)
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)


    def forward(self, inputs):
        inputs = quantize_active_overflow(inputs, self.bitA)  # Do not remove

        outputs = tl.nn.conv2d(
            input=inputs, filters=self.W, strides=self.strides, padding=self.padding, data_format=self.data_format,
            dilations=self._dilation_rate, name=self.name
        )
        para_bn_shape = outputs.get_shape()[-1:]
        if self.gamma_init:
            self.scale_para = self._get_weights("scale_para", shape=para_bn_shape, init=self.gamma_init)
        else:
            self.scale_para = None

        if self.beta_init:
            self.offset_para = self._get_weights("offset_para", shape=para_bn_shape, init=self.beta_init)
        else:
            self.offset_para = None
        moving_mean = self._get_weights(
            "moving_mean", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )
        moving_variance = self._get_weights(
            "moving_variance", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )
        mean, variance = tf.nn.moments(outputs, axes=list(range(len(outputs.get_shape()) - 1)))

        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, self.decay, zero_debias=False)  # if zero_debias=True, has bias
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, mean, self.decay, zero_debias=False)  # if zero_debias=True, has bias
        if self.is_train:
            mean, var = self.mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance)
        else:
            mean, var = moving_mean, moving_variance
        w_fold = self._w_fold(self.W, self.scale_para, var, self.epsilon)
        bias_fold = self._bias_fold(self.offset_para, self.scale_para, mean, var, self.epsilon)

        W_ = quantize_weight_overflow(w_fold, self.bitW)
        conv_fold = tf.nn.conv2d(
            inputs, W_, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        conv_fold = tf.nn.bias_add(conv_fold, bias_fold, name='bn_bias_add')
        if self.act:
            conv_fold = self.act(conv_fold)
            print(conv_fold)
        return conv_fold


    def mean_var_with_update(self, update_moving_mean, update_moving_variance, mean, variance):
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    def _bias_fold(self, beta, gama, mean, var, epsilon):
        return tf.subtract(beta, tf.compat.v1.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))
