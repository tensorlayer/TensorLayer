#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# from tensorlayer.layers.core import LayersConfig

__all__ = [
    'Conv1dLayer',
    'Conv2dLayer',
    'Conv3dLayer',
]


class Conv1dLayer(Layer):
    """
    The :class:`Conv1dLayer` class is a 1D CNN layer, see `tf.nn.conv1d <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv1d>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_length, in_channels, out_channels).
    stride : int
        The number of entries by which the filter is moved right at a step.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        'NWC' or 'NCW', Default is 'NWC' as it is a 1D CNN.
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    name : None or str
        A unique layer name

    Notes
    -----
    - shape = [w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 100, 1], name='input')
    >>> conv1d = tl.layers.Conv1dLayer(shape=(5, 1, 32), stride=2, b_init=None, name='conv1d_1')
    >>> print(conv1d)
    >>> tensor = tl.layers.Conv1dLayer(shape=(5, 1, 32), stride=2, act=tf.nn.relu, name='conv1d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            act=None,
            shape=(5, 1, 5),
            stride=1,
            padding='SAME',
            data_format='NWC',
            dilation_rate=1,
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'cnn1d_layer',
    ):
        super().__init__(name, act=act)
        self.n_filter = shape[-1]
        self.filter_size = shape[0]
        self.shape = shape
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = shape[-2]

        self.build(None)
        self._built = True

        logging.info(
            "Conv1dLayer %s: shape: %s stride: %s pad: %s act: %s" % (
                self.name, str(shape), str(stride), padding,
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
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init)

    def forward(self, inputs):

        outputs = tf.nn.conv1d(
            input=inputs,
            filters=self.W,
            stride=self.stride,
            padding=self.padding,
            dilations=[
                self.dilation_rate,
            ],
            data_format=self.data_format,
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv2d>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    dilation_rate : tuple of int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    name : None or str
        A unique layer name.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 28, 28, 1], name='input')
    >>> conv2d = tl.layers.Conv2dLayer(shape=(5, 5, 1, 32), strides=(1, 1, 1, 1), b_init=None, name='conv2d_1')
    >>> print(conv2d)
    >>> tensor = tl.layers.Conv2dLayer(shape=(5, 5, 1, 32), strides=(1, 1, 1, 1), act=tf.nn.relu, name='conv2d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            act=None,
            shape=(5, 5, 1, 100),
            strides=(1, 1, 1, 1),
            padding='SAME',
            data_format='NHWC',
            dilation_rate=(1, 1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'cnn2d_layer',
    ):
        super().__init__(name, act=act)
        self.n_filter = shape[-1]
        self.filter_size = (shape[0], shape[1])
        self.shape = shape
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = shape[-2]

        self.build(None)
        self._built = True

        logging.info(
            "Conv2dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation_rate != [
                1,
        ] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=list(self.dilation_rate),
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv3d>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
        Must be in the same order as the shape dimension.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "NDHWC" or "NCDHW", default is "NDHWC".
    dilation_rate : tuple of int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    name : None or str
        A unique layer name.

    Notes
    -----
    - shape = [d, h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([8, 100, 100, 100, 3], name='input')
    >>> conv3d = tl.layers.Conv3dLayer(shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1), b_init=None, name='conv3d_1')
    >>> print(conv3d)
    >>> tensor = tl.layers.Conv3dLayer(shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1), act=tf.nn.relu, name='conv3d_2')(net)
    >>> print(tensor)

    """

    def __init__(
            self,
            act=None,
            shape=(2, 2, 2, 3, 32),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            data_format='NDHWC',
            dilation_rate=(1, 1, 1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'cnn3d_layer'
    ):
        super().__init__(name, act=act)
        self.n_filter = shape[-1]
        self.filter_size = (shape[0], shape[1], shape[2])
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = shape[-2]

        self.build(None)
        self._built = True

        logging.info(
            "Conv3dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, padding={padding}'
        )
        if self.dilation_rate != [
                1,
        ] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):

        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv3d(
            input=inputs,
            filters=self.W,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,  #'NDHWC',
            dilations=list(self.dilation_rate),  #[1, 1, 1, 1, 1],
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
