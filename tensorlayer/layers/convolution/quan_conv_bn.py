#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module
from tensorflow.python.training import moving_averages
from tensorlayer.layers.utils import (quantize_active_overflow, quantize_weight_overflow)
from tensorlayer.backend import BACKEND

__all__ = ['QuanConv2dWithBN']


class QuanConv2dWithBN(Module):
    """The :class:`QuanConv2dWithBN` class is a quantized convolutional layer with BN, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would keep the same.

    Parameters
    ----------
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
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> net = tl.layers.Input([50, 256, 256, 3])
    >>> layer = tl.layers.QuanConv2dWithBN(n_filter=64, filter_size=(5,5),strides=(1,1),padding='SAME',name='qcnnbn1')
    >>> print(layer)
    >>> net = tl.layers.QuanConv2dWithBN(n_filter=64, filter_size=(5,5),strides=(1,1),padding='SAME',name='qcnnbn1')(net)
    >>> print(net)
    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        act=None,
        decay=0.9,
        epsilon=1e-5,
        is_train=False,
        gamma_init=tl.initializers.truncated_normal(stddev=0.02),
        beta_init=tl.initializers.truncated_normal(stddev=0.02),
        bitW=8,
        bitA=8,
        use_gemm=False,
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        W_init_args=None,
        data_format="channels_last",
        dilation_rate=(1, 1),
        in_channels=None,
        name='quan_cnn2d_bn',
    ):
        super(QuanConv2dWithBN, self).__init__(act=act, name=name)
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
        self.dilation_rate = dilation_rate
        self.in_channels = in_channels
        logging.info(
            "QuanConv2dWithBN %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s " % (
                self.name, n_filter, filter_size, str(strides), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

        if BACKEND == 'mindspore':
            raise NotImplementedError("MindSpore backend does not implement this method")

        if self.in_channels:
            self.build(None)
            self._built = True

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}' + actstr
        )
        if self.dilation_rate != (1, ) * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self.strides[0], self.strides[1], 1]
            self._dilation_rate = [1, self.dilation_rate[0], self.dilation_rate[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self.strides[0], self.strides[1]]
            self._dilation_rate = [1, 1, self.dilation_rate[0], self.dilation_rate[1]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.filter_size[0], self.filter_size[1], self.in_channels, self.n_filter)
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        para_bn_shape = (self.n_filter, )
        if self.gamma_init:
            self.scale_para = self._get_weights(
                "scale_para", shape=para_bn_shape, init=self.gamma_init, trainable=self.is_train
            )
        else:
            self.scale_para = None

        if self.beta_init:
            self.offset_para = self._get_weights(
                "offset_para", shape=para_bn_shape, init=self.beta_init, trainable=self.is_train
            )
        else:
            self.offset_para = None

        self.moving_mean = self._get_weights(
            "moving_mean", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )
        self.moving_variance = self._get_weights(
            "moving_variance", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        x = inputs
        inputs = quantize_active_overflow(inputs, self.bitA)  # Do not remove
        outputs = tf.nn.conv2d(
            input=x, filters=self.W, strides=self._strides, padding=self.padding, data_format=self.data_format,
            dilations=self._dilation_rate, name=self.name
        )

        mean, variance = tf.nn.moments(outputs, axes=list(range(len(outputs.get_shape()) - 1)))

        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay, zero_debias=False
        )  # if zero_debias=True, has bias
        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, mean, self.decay, zero_debias=False
        )  # if zero_debias=True, has bias

        if self.is_train:
            mean, var = self.mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance)
        else:
            mean, var = self.moving_mean, self.moving_variance

        w_fold = self._w_fold(self.W, self.scale_para, var, self.epsilon)

        W_ = quantize_weight_overflow(w_fold, self.bitW)

        conv_fold = tf.nn.conv2d(inputs, W_, strides=self.strides, padding=self.padding, data_format=self.data_format)

        if self.beta_init:
            bias_fold = self._bias_fold(self.offset_para, self.scale_para, mean, var, self.epsilon)
            conv_fold = tf.nn.bias_add(conv_fold, bias_fold, name='bn_bias_add')

        if self.act:
            conv_fold = self.act(conv_fold)

        return conv_fold

    def mean_var_with_update(self, update_moving_mean, update_moving_variance, mean, variance):
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    def _w_fold(self, w, gama, var, epsilon):
        return tf.compat.v1.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))

    def _bias_fold(self, beta, gama, mean, var, epsilon):
        return tf.subtract(beta, tf.compat.v1.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))
