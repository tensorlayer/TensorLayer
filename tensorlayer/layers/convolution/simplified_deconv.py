#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import get_collection_trainable

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
    >>> tensor = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d_2')(net)
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
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            # dtype=tf.float32,
            name=self.name,
        )
        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]
        _out = self.layer(
            tf.convert_to_tensor(np.random.uniform(size=inputs_shape), dtype=np.float32)
        )  #np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        outputs_shape = _out.shape
        self._trainable_weights = self.layer.weights

    def forward(self, inputs):
        outputs = self.layer(inputs)
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

    >>> net = tl.layers.Input([5, 100, 100, 100, 32], name='input')
    >>> deconv3d = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), in_channels=32, name='DeConv3d_1')
    >>> print(deconv3d)
    >>> tensor = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), name='DeConv3d_2')(net)
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
        self.in_channels = in_channels,

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
        self.layer = tf.keras.layers.Conv3DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            name=self.name,
        )
        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]

        _out = self.layer(
            tf.convert_to_tensor(np.random.uniform(size=inputs_shape), dtype=np.float32)
        )  #self.layer(np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        outputs_shape = _out.shape
        # self._add_weights(self.layer.weights)
        self._trainable_weights = self.layer.weights

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs
