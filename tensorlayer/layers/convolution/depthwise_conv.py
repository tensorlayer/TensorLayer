#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

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

    """

    # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py
    def __init__(
            self,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None, #'depthwise_conv2d',
    ):
        # super(DepthwiseConv2d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.stride = stride
        self.act = act
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        logging.info(
            "DepthwiseConv2d %s: filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, str(filter_size), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs):
        self.pre_channel = inputs.shape.as_list[-1]
        if self.pre_channel is None: # if pre_channel is ?, it happens when using Spatial Transformer Net
            self.pre_channel = 1
            logging.info("[warnings] unknown input channels, set to 1")

        self.filter_size = [self.filter_size[0], self.filter_size[1], self.pre_channel, self.depth_multiplier]

        if len(self.strides) == 2:
            self.strides = [1, self.strides[0], self.strides[1], 1]

        if len(self.strides) != 4:
            raise AssertionError("len(strides) should be 4.")

        self.W = tf.get_variable(
            name=self.name+'\W_depthwise2d', shape=self.filter_size, initializer=self.W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
        )  # [filter_height, filter_width, in_channels, depth_multiplier]

        if self.b_init:
            self.b = tf.get_variable(
                name=self.name+'\b_depthwise2d', shape=(self.pre_channel * self.depth_multiplier), initializer=self.b_init,
                dtype=LayersConfig.tf_dtype, **self.b_init_args
            )
            self.add_weights([self.W, self.b])
        else:
            self.add_weights(self.W)

    def forward(self, inputs):

        outputs = tf.nn.depthwise_conv2d(inputs, self.W, strides=self.strides, padding=self.padding, rate=self.dilation_rate)
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        outputs = self.act(outputs)
