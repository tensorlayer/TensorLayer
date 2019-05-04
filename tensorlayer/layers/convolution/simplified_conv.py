#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import get_collection_trainable

__all__ = [
    'Conv1d',
    'Conv2d',
    'Conv3d',
]


class Conv1d(Layer):
    """Simplified version of :class:`Conv1dLayer`.

    Parameters
    ----------
    n_filter : int
        The number of filters
    filter_size : int
        The filter size
    stride : int
        The stride step
    dilation_rate : int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The function that is applied to the layer activations
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channel_last" (NWC, default) or "channels_first" (NCW).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 100, 1], name='input')
    >>> conv1d = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2, b_init=None, in_channels=1, name='conv1d_1')
    >>> print(conv1d)
    >>> tensor = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2, act=tf.nn.relu, name='conv1d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            n_filter=32,
            filter_size=5,
            stride=1,
            act=None,
            padding='SAME',
            data_format="channels_last",
            dilation_rate=1,
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None  # 'conv1d'
    ):
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.stride = stride
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv1d %s: n_filter: %d filter_size: %s stride: %d pad: %s act: %s" % (
                self.name, n_filter, filter_size, stride, padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', stride={stride}, padding={padding}'
        )
        if self.dilation_rate != 1:
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (self.filter_size, self.in_channels, self.n_filter)

        # TODO : check
        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv1d(
            input=inputs,
            filters=self.W,
            stride=self.stride,
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


class Conv2d(Layer):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 400, 400, 3], name='input')
    >>> conv2d = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), stride=(2, 2), b_init=None, in_channels=3, name='conv2d_1')
    >>> print(conv2d)
    >>> tensor = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), stride=(2, 2), act=tf.nn.relu, name='conv2d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None  # 'conv2d',
    ):
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self._strides = self.strides = strides
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

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

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,  #'NHWC',
            dilations=self._dilation_rate,  #[1, 1, 1, 1],
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


class Conv3d(Layer):
    """Simplified version of :class:`Conv3dLayer`.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NDHWC, default) or "channels_first" (NCDHW).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 20, 20, 20, 3], name='input')
    >>> conv3d = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3, 3), stride=(2, 2, 2), b_init=None, in_channels=3, name='conv3d_1')
    >>> print(conv3d)
    >>> tensor = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3, 3), stride=(2, 2, 2), act=tf.nn.relu, name='conv3d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3, 3),
            strides=(1, 1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None  # 'conv3d',
    ):
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self._strides = self.strides = strides
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "Conv3d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
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
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
            if self.in_channels is None:
                self.in_channels = inputs_shape[-1]
            self._strides = [1, self._strides[0], self._strides[1], self._strides[2], 1]
            self._dilation_rate = [1, self.dilation_rate[0], self.dilation_rate[1], self.dilation_rate[2], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
            if self.in_channels is None:
                self.in_channels = inputs_shape[1]
            self._strides = [1, 1, self._strides[0], self._strides[1], self._strides[2]]
            self._dilation_rate = [1, 1, self._dilation_rate[0], self._dilation_rate[1], self._dilation_rate[2]]
        else:
            raise Exception("data_format should be either channels_last or channels_first")

        self.filter_shape = (
            self.filter_size[0], self.filter_size[1], self.filter_size[2], self.in_channels, self.n_filter
        )

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv3d(
            input=inputs,
            filters=self.W,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,  #'NDHWC',
            dilations=self._dilation_rate,  #[1, 1, 1, 1, 1],
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
