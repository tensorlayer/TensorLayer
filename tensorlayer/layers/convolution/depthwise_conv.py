#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# from tensorlayer.layers.core import LayersConfig

__all__ = [
    'DepthwiseConv2d',
]


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    filter_size : tuple of 2 int
        The filter size (height, width).
    strides : tuple of 2 int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation_rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([8, 200, 200, 32], name='input')
    >>> depthwiseconv2d = tl.layers.DepthwiseConv2d(
    ...     filter_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise'
    ... )(net)
    >>> print(depthwiseconv2d)
    >>> output shape : (8, 200, 200, 64)


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
            data_format='channels_last',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None  # 'depthwise_conv2d'
    ):
        super().__init__(name, act=act)
        self.filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.dilation_rate = self._dilation_rate = dilation_rate
        self.data_format = data_format
        self.depth_multiplier = depth_multiplier
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "DepthwiseConv2d %s: filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, str(filter_size), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation_rate != (1, ) * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, n_filter=self.in_channels * self.depth_multiplier, **self.__dict__
        )

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

        self.filter_shape = (self.filter_size[0], self.filter_size[1], self.in_channels, self.depth_multiplier)

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.in_channels * self.depth_multiplier), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=self.W,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilation_rate,
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
