#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl

import tensorflow as tf
from tensorlayer import logging
from tensorlayer.layers.core import Layer

__all__ = [
    # 'DeConv1d'  # TODO: Shall be implemented
    'DeConv2d',
    'DeConv3d',
]


class DeConv2d(Layer):
    """Simplified version of :class:`DeConv2dLayer`, see `tf.nn.conv3d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv2d_transpose>`__.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The stride step (height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation_rate : int of tuple of int
        The dilation rate to use for dilated convolution
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([5, 100, 100, 32], name='input')
    >>> deconv2d = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), in_channels=32, name='DeConv2d_1')
    >>> print(deconv2d)
    >>> tensor = tl.layers.DeConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), name='DeConv2d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(2, 2),
        act=None,
        padding='SAME',
        dilation_rate=(1, 1),
        data_format='channels_last',
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None  # 'decnn2d'
    ):
        super().__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        # Attention: To build, we need not only the in_channels!
        # if self.in_channels:
        #     self.build(None)
        #     self._built = True

        logging.info(
            "DeConv2d {}: n_filters: {} strides: {} padding: {} act: {} dilation: {}".format(
                self.name,
                str(n_filter),
                str(strides),
                padding,
                self.act.__name__ if self.act is not None else 'No Activation',
                dilation_rate,
            )
        )

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2, DeConv2d and DeConv2dLayer are different.")

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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        kernel_h, kernel_w = self.filter_size
        stride_h, stride_w = self.strides
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self.strides[0], self.strides[1], 1]
            self._dilation_rate = [1, self.dilation_rate[0], self.dilation_rate[1], 1]
            H = deconv_length(input_length=inputs_shape[1], stride=stride_h, filter_size=kernel_h, padding=self.padding)
            W = deconv_length(input_length=inputs_shape[2], stride=stride_w, filter_size=kernel_w, padding=self.padding)
            self.outshape = [inputs_shape[0], H, W, self.n_filter]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self.strides[0], self.strides[1]]
            self._dilation_rate = [1, 1, self.dilation_rate[0], self.dilation_rate[1]]
            H = deconv_length(input_length=inputs_shape[2], stride=stride_h, filter_size=kernel_h, padding=self.padding)
            W = deconv_length(input_length=inputs_shape[3], stride=stride_w, filter_size=kernel_w, padding=self.padding)
            self.outshape = [inputs_shape[0], self.n_filter, H, W]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.filter_size[0], self.filter_size[1], self.n_filter, self.in_channels)

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)

        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]

    def forward(self, inputs):
        outputs = tf.nn.conv2d_transpose(
            input=inputs, filters=self.W, output_shape=self.outshape, strides=self._strides, padding=self.padding,
            data_format=self.data_format, dilations=self._dilation_rate, name=self.name
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


class DeConv3d(Layer):
    """Simplified version of :class:`DeConv3dLayer`, see `tf.nn.conv3d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv3d_transpose>`__.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (depth, height, width).
    strides : tuple of int
        The stride step (depth, height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    data_format : str
        "channels_last" (NDHWC, default) or "channels_first" (NCDHW).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([5, 10, 10, 10, 32], name='input')
    >>> deconv3d = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), in_channels=32, name='DeConv3d_1')
    >>> print(deconv3d)
    >>> tensor = tl.layers.DeConv3d(n_filter=64, filter_size=(3, 3, 3), strides=(2, 2, 2), name='DeConv3d_2')(net)
    >>> print(tensor)

    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding='SAME',
        act=None,
        data_format='channels_last',
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None  # 'decnn3d'
    ):
        super().__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        # Attention: To build, we need not only the in_channels!
        # if self.in_channels:
        #     self.build(None)
        #     self._built = True

        logging.info(
            "DeConv3d %s: n_filters: %s strides: %s pad: %s act: %s" % (
                self.name, str(n_filter), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        if len(strides) != 3:
            raise ValueError("len(strides) should be 3, DeConv3d and DeConv3dLayer are different.")

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}'
        )
        # if self.dilation_rate != (1,) * len(self.dilation_rate):
        #     s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        kernel_d, kernel_h, kernel_w = self.filter_size
        stride_d, stride_h, stride_w = self.strides
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]
            D = deconv_length(input_length=inputs_shape[1], stride=stride_d, filter_size=kernel_d, padding=self.padding)
            H = deconv_length(input_length=inputs_shape[2], stride=stride_h, filter_size=kernel_h, padding=self.padding)
            W = deconv_length(input_length=inputs_shape[3], stride=stride_w, filter_size=kernel_w, padding=self.padding)
            self.outshape = [inputs_shape[0], D, H, W, self.n_filter]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self.strides[0], self.strides[1], self.strides[2]]
            D = deconv_length(input_length=inputs_shape[2], stride=stride_d, filter_size=kernel_d, padding=self.padding)
            H = deconv_length(input_length=inputs_shape[3], stride=stride_h, filter_size=kernel_h, padding=self.padding)
            W = deconv_length(input_length=inputs_shape[4], stride=stride_w, filter_size=kernel_w, padding=self.padding)
            self.outshape = [inputs_shape[0], self.n_filter, D, H, W]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (
            self.filter_size[0], self.filter_size[1], self.filter_size[2], self.n_filter, self.in_channels
        )

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv3d_transpose(
            input=inputs, filters=self.W, output_shape=self.outshape, strides=self._strides, padding=self.padding,
            data_format=self.data_format, name=self.name
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


def deconv_length(input_length, filter_size, padding, stride):
    """Determines output length of a transposed convolution given input length.
    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same, SAME", "valid, VALID"
        stride: integer.
    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    if padding in ['valid', 'VALID']:
        input_length = input_length * stride + max(filter_size - stride, 0)
    elif padding in ['same', 'SAME']:
        input_length = input_length * stride
    else:
        raise Exception("Unsupported padding: {}".format(padding))
    return input_length
