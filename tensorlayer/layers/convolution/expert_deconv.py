#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# from tensorlayer.layers.core import LayersConfig

__all__ = [
    'DeConv1dLayer',
    'DeConv2dLayer',
    'DeConv3dLayer',
]


class DeConv1dLayer(Layer):
    """A de-convolution 1D layer.

    See `tf.nn.conv1d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv1d_transpose>`__.

    Parameters
    ----------
    act : activation function or None
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (height, width, output_channels, in_channels).
        The filter's ``in_channels`` dimension must match that of value.
    outputs_shape : tuple of int
        Output shape of the deconvolution,
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "NWC" or "NCW", default is "NWC".
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    name : None or str
        A unique layer name.

    Notes
    -----
    - shape = [w, the number of output channels of this layer, the number of output channel of the previous layer].
    - outputs_shape = [batch_size, any, the number of output channels of this layer].
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> input_layer = Input([8, 25, 32], name='input_layer')
    >>> deconv1d = tl.layers.DeConv1dLayer(
    ...     shape=(5, 64, 32), outputs_shape=(8, 50, 64), strides=(1, 2, 1), name='deconv1dlayer'
    ... )
    >>> print(deconv1d)
    >>> tensor = tl.layers.DeConv1dLayer(
    ...     shape=(5, 64, 32), outputs_shape=(8, 50, 64), strides=(1, 2, 1), name='deconv1dlayer'
    ... )(input_layer)
    >>> print(tensor)
    >>> output shape : (8, 50, 64)

    """

    def __init__(
            self,
            act=None,
            shape=(3, 128, 256),
            outputs_shape=(1, 256, 128),
            strides=(1, 2, 1),
            padding='SAME',
            data_format='NWC',
            dilation_rate=(1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'decnn1d_layer',
    ):
        super().__init__(name)
        self.act = act
        self.shape = shape
        self.outputs_shape = outputs_shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = self.shape[-1]

        self.build(None)
        self._built = True

        logging.info(
            "DeConv1dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(outputs_shape), str(strides), padding,
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
            classname=self.__class__.__name__, n_filter=self.shape[-2], filter_size=self.shape[0], **self.__dict__
        )

    def build(self, inputs):
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.shape[-2]), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv1d_transpose(
            input=inputs,
            filters=self.W,
            output_shape=self.outputs_shape,
            strides=list(self.strides),
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


class DeConv2dLayer(Layer):
    """A de-convolution 2D layer.

    See `tf.nn.conv2d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv2d_transpose>`__.

    Parameters
    ----------
    act : activation function or None
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (height, width, output_channels, in_channels).
        The filter's ``in_channels`` dimension must match that of value.
    outputs_shape : tuple of int
        Output shape of the deconvolution,
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
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
    - shape = [h, w, the number of output channels of this layer, the number of output channel of the previous layer].
    - outputs_shape = [batch_size, any, any, the number of output channels of this layer].
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    TODO: Add the example code of a part of the generator in DCGAN example

    U-Net

    >>> ....
    >>> conv10 = tl.layers.Conv2dLayer(
    ...        act=tf.nn.relu,
    ...        shape=(3, 3, 1024, 1024), strides=(1, 1, 1, 1), padding='SAME',
    ...        W_init=w_init, b_init=b_init, name='conv10'
    ... )(conv9)
    >>> print(conv10)
    (batch_size, 32, 32, 1024)
    >>> deconv1 = tl.layers.DeConv2dLayer(
    ...         act=tf.nn.relu,
    ...         shape=(3, 3, 512, 1024), strides=(1, 2, 2, 1), outputs_shape=(batch_size, 64, 64, 512),
    ...         padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1'
    ... )(conv10)

    """

    def __init__(
            self,
            act=None,
            shape=(3, 3, 128, 256),
            outputs_shape=(1, 256, 256, 128),
            strides=(1, 2, 2, 1),
            padding='SAME',
            data_format='NHWC',
            dilation_rate=(1, 1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'decnn2d_layer',
    ):
        super().__init__(name)
        self.act = act
        self.shape = shape
        self.outputs_shape = outputs_shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = self.shape[-1]

        self.build(None)
        self._built = True

        logging.info(
            "DeConv2dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(outputs_shape), str(strides), padding,
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
            classname=self.__class__.__name__, n_filter=self.shape[-2], filter_size=(self.shape[0], self.shape[1]),
            **self.__dict__
        )

    def build(self, inputs):
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.shape[-2]), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv2d_transpose(
            input=inputs,
            filters=self.W,
            output_shape=self.outputs_shape,
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


class DeConv3dLayer(Layer):
    """A de-convolution 3D layer.

    See `tf.nn.conv3d_transpose <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/conv3d_transpose>`__.

    Parameters
    ----------
    act : activation function or None
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (depth, height, width, output_channels, in_channels).
        The filter's in_channels dimension must match that of value.
    outputs_shape : tuple of int
        The output shape of the deconvolution.
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
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
    - shape = [d, h, w, the number of output channels of this layer, the number of output channel of the previous layer].
    - outputs_shape = [batch_size, any, any, any, the number of output channels of this layer].
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> input_layer = Input([8, 10, 10, 10 32], name='input_layer')
    >>> deconv3d = tl.layers.DeConv3dLayer(
    ...     shape=(2, 2, 2, 128, 32), outputs_shape=(8, 20, 20, 20, 128), strides=(1, 2, 2, 2, 1), name='deconv3dlayer'
    ... )
    >>> print(deconv3d)
    >>> tensor = tl.layers.DeConv1dLayer(
    ...     shape=(2, 2, 2, 128, 32), outputs_shape=(8, 20, 20, 20, 128), strides=(1, 2, 2, 2, 1), name='deconv3dlayer'
    ... )(input_layer)
    >>> print(tensor)
    >>> output shape : (8, 20, 20, 20, 128)

    """

    def __init__(
            self,
            act=None,
            shape=(2, 2, 2, 128, 256),
            outputs_shape=(1, 12, 32, 32, 128),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            data_format='NDHWC',
            dilation_rate=(1, 1, 1, 1, 1),
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            name=None  # 'decnn3d_layer',
    ):
        super().__init__(name)
        self.act = act
        self.shape = shape
        self.outputs_shape = outputs_shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = self.shape[-1]

        self.build(None)
        self._built = True

        logging.info(
            "DeConv3dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(outputs_shape), str(strides), padding,
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
            classname=self.__class__.__name__, n_filter=self.shape[-2],
            filter_size=(self.shape[0], self.shape[1], self.shape[2]), **self.__dict__
        )

    def build(self, inputs):
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.shape[-2]), init=self.b_init)

    def forward(self, inputs):
        outputs = tf.nn.conv3d_transpose(
            input=inputs, filters=self.W, output_shape=self.outputs_shape, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=list(self.dilation_rate), name=self.name
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
