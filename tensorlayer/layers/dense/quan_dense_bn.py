#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
# from tensorlayer.layers.core import LayersConfig
from tensorflow.python.training import moving_averages

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import (quantize_active_overflow, quantize_weight_overflow)

__all__ = [
    'QuanDenseWithBN',
]


class QuanDenseWithBN(Layer):
    """The :class:`QuanDenseWithBN` class is a quantized fully connected layer with BN, which weights are 'bitW' bits and the output of the previous layer
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
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : a str
        A unique layer name.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> net = tl.layers.Input([50, 256])
    >>> layer = tl.layers.QuanDenseWithBN(128, act='relu', name='qdbn1')(net)
    >>> print(layer)
    >>> net = tl.layers.QuanDenseWithBN(256, act='relu', name='qdbn2')(net)
    >>> print(net)
    """

    def __init__(
        self,
        n_units=100,
        act=None,
        decay=0.9,
        epsilon=1e-5,
        is_train=False,
        bitW=8,
        bitA=8,
        gamma_init=tl.initializers.truncated_normal(stddev=0.05),
        beta_init=tl.initializers.truncated_normal(stddev=0.05),
        use_gemm=False,
        W_init=tl.initializers.truncated_normal(stddev=0.05),
        W_init_args=None,
        in_channels=None,
        name=None,  # 'quan_dense_with_bn',
    ):
        super(QuanDenseWithBN, self).__init__(act=act, W_init_args=W_init_args, name=name)
        self.n_units = n_units
        self.decay = decay
        self.epsilon = epsilon
        self.is_train = is_train
        self.bitW = bitW
        self.bitA = bitA
        self.gamma_init = gamma_init
        self.beta_init = beta_init
        self.use_gemm = use_gemm
        self.W_init = W_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info(
            "QuanDenseLayerWithBN  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        s += ', bitW={bitW}, bitA={bitA}'
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if self.in_channels is None:
            self.in_channels = inputs_shape[1]

        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = inputs_shape[-1]
        self.W = self._get_weights("weights", shape=(n_in, self.n_units), init=self.W_init)

        para_bn_shape = (self.n_units, )
        if self.gamma_init:
            self.scale_para = self._get_weights("gamm_weights", shape=para_bn_shape, init=self.gamma_init)
        else:
            self.scale_para = None

        if self.beta_init:
            self.offset_para = self._get_weights("beta_weights", shape=para_bn_shape, init=self.beta_init)
        else:
            self.offset_para = None

        self.moving_mean = self._get_weights(
            "moving_mean", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )
        self.moving_variance = self._get_weights(
            "moving_variacne", shape=para_bn_shape, init=tl.initializers.constant(1.0), trainable=False
        )

    def forward(self, inputs):
        x = inputs
        inputs = quantize_active_overflow(inputs, self.bitA)
        mid_out = tf.matmul(x, self.W)

        mean, variance = tf.nn.moments(x=mid_out, axes=list(range(len(mid_out.get_shape()) - 1)))

        update_moving_mean = moving_averages.assign_moving_average(
            self.moving_mean, mean, self.decay, zero_debias=False
        )  # if zero_debias=True, has bias

        update_moving_variance = moving_averages.assign_moving_average(
            self.moving_variance, variance, self.decay, zero_debias=False
        )  # if zero_debias=True, has bias

        if self.is_train:
            mean, var = self.mean_var_with_update(update_moving_mean, update_moving_variance, mean, variance)
        else:
            mean, var = self.moving_mean, self.moving_variance

        w_fold = self._w_fold(self.W, self.scale_para, var, self.epsilon)

        W = quantize_weight_overflow(w_fold, self.bitW)

        outputs = tf.matmul(inputs, W)

        if self.beta_init:
            bias_fold = self._bias_fold(self.offset_para, self.scale_para, mean, var, self.epsilon)
            outputs = tf.nn.bias_add(outputs, bias_fold, name='bias_add')
        else:
            outputs = outputs

        if self.act:
            outputs = self.act(outputs)
        else:
            outputs = outputs
        return outputs

    def mean_var_with_update(self, update_moving_mean, update_moving_variance, mean, variance):
        with tf.control_dependencies([update_moving_mean, update_moving_variance]):
            return tf.identity(mean), tf.identity(variance)

    def _w_fold(self, w, gama, var, epsilon):
        return tf.compat.v1.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))

    def _bias_fold(self, beta, gama, mean, var, epsilon):
        return tf.subtract(beta, tf.compat.v1.div(tf.multiply(gama, mean), tf.sqrt(var + epsilon)))
