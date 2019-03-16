#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
            name='cnn1d_layer',
    ):
        super().__init__(name)
        self.act = act
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
        self.name = name

        logging.info(
            "Conv1dLayer %s: shape: %s stride: %s pad: %s act: %s" % (
                self.name, str(shape), str(stride), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
             ', stride={stride}, padding={padding}')
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
        self.W = self._get_weights(
            "filters", shape=self.shape, init=self.W_init
        )
        if self.b_init:
            self.b = self._get_weights(
                "biases",
                shape=(self.n_filter),  #self.shape[-1]),
                init=self.b_init
            )

    def forward(self, inputs):

        outputs = tf.nn.conv1d(
            input=inputs,
            filters=self.W,
            stride=self.stride,
            padding=self.padding,
            dilations=(self.dilation_rate, ),
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
    dilation_rate : list of int
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

    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = tl.layers.Input(x, name='input_layer')
    >>> net = tl.layers.Conv2dLayer(net,
    ...                   act = tf.nn.relu,
    ...                   shape = (5, 5, 1, 32),  # 32 features for each 5x5 patch
    ...                   strides = (1, 1, 1, 1),
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> net = tl.layers.Pool(net,
    ...                   ksize=(1, 2, 2, 1),
    ...                   strides=(1, 2, 2, 1),
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    Without TensorLayer, you can implement 2D convolution as follow.

    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )

    """

    def __init__(
            self,
            act=None,
            shape=(5, 5, 1, 100),
            strides=(1, 1, 1, 1),
            padding='SAME',
            data_format=None,
            dilation_rate=[1, 1, 1, 1],
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name='cnn2d_layer',
    ):
        super().__init__(name)
        self.act = act
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
        self.name = name

        logging.info(
            "Conv2dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.dilation_rate != [1,] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):
        self.W = self._get_weights(
            "filters", shape=self.shape, init=self.W_init
        )
        if self.b_init:
            self.b = self._get_weights(
                "biases", shape=(self.n_filter), init=self.b_init
            )

    def forward(self, inputs):
        outputs = tf.nn.conv2d(
            input=inputs,
            filters=self.W,
            strides=self.strides,
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
    dilation_rate : list of int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    name : None or str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
    >>> n = tl.layers.Input(x, name='in3')
    >>> n = tl.layers.Conv3dLayer(n, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
    [None, 50, 50, 50, 32]
    """

    def __init__(
            self,
            act=None,
            shape=(2, 2, 2, 3, 32),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            data_format='NDHWC',
            dilation_rate=[1, 1, 1, 1, 1],
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name='cnn3d_layer'
    ):
        super().__init__(name)
        self.act = act
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
        self.name = name

        logging.info(
            "Conv3dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.dilation_rate != [1,] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs):

        self.W = self._get_weights(
            "filters", shape=self.shape, init=self.W_init
        )
        if self.b_init:
            self.b = self._get_weights(
                "biases", shape=(self.n_filter), init=self.b_init
            )

    def forward(self, inputs):
        outputs = tf.nn.conv3d(
            input=inputs,
            filters=self.W,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,  #'NDHWC',
            dilations=self.dilation_rate,  #[1, 1, 1, 1, 1],
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
