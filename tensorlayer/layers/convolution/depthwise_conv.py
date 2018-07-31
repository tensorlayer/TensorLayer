#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

__all__ = [
    'DepthwiseConv2d',
]


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    filter_size : tuple of int
        The filter size (height, width).
    stride : tuple of int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    dilation_rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = InputLayer(x, name='input')
    >>> net = Conv2d(net, 32, (3, 3), (2, 2), b_init=None, name='cin')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bnin')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (1, 1), b_init=None, name='cdw1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn11')
    >>> net = Conv2d(net, 64, (1, 1), (1, 1), b_init=None, name='c1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn12')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (2, 2), b_init=None, name='cdw2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn21')
    >>> net = Conv2d(net, 128, (1, 1), (1, 1), b_init=None, name='c2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn22')

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """ # # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            shape=(3, 3),
            strides=(1, 1),
            padding='SAME',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            act=None,
            name='depthwise_conv2d',
    ):

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        elif len(strides) != 4:
            raise ValueError("len(strides) should be 4.")

        self.prev_layer = prev_layer
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(DepthwiseConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("shape: %s" % str(self.shape))
        except AttributeError:
            pass

        try:
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(DepthwiseConv2d, self).__call__(prev_layer)

        try:
            input_channels = int(self.inputs.get_shape()[-1])

        except TypeError:  # if input_channels is ?, it happens when using Spatial Transformer Net
            input_channels = 1
            logging.warning("unknown input channels, set to 1")

        w_shape = [self.shape[0], self.shape[1], input_channels, self.depth_multiplier]

        with tf.variable_scope(self.name):

            W = self._get_tf_variable(
                name='W_depthwise2d', shape=w_shape, initializer=self.W_init, dtype=self.inputs.dtype,
                **self.W_init_args
            )

            self.outputs = tf.nn.depthwise_conv2d(
                self.inputs, W, strides=self.strides, padding=self.padding,  rate=self.dilation_rate
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_depthwise2d', shape=(input_channels * self.depth_multiplier), initializer=self.b_init,
                    dtype=self.inputs.dtype, **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)




