#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

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
            prev_layer,
            shape=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='depthwise_conv2d',
    ):
        super(DepthwiseConv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DepthwiseConv2d %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknown input channels, set to 1")

        shape = [shape[0], shape[1], pre_channel, depth_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        if len(strides) != 4:
            raise AssertionError("len(strides) should be 4.")

        with tf.variable_scope(name):

            W = tf.get_variable(
                name='W_depthwise2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )  # [filter_height, filter_width, in_channels, depth_multiplier]

            self.outputs = tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=dilation_rate)

            if b_init:
                b = tf.get_variable(
                    name='b_depthwise2d', shape=(pre_channel * depth_multiplier), initializer=b_init,
                    dtype=LayersConfig.tf_dtype, **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:
            self._add_params(W)
