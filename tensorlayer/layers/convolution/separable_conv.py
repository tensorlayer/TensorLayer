#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

import numpy as np

from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'SeparableConv1d',
    'SeparableConv2d',
]


class SeparableConv1d(Layer):
    """The :class:`SeparableConv1d` class is a 1D depthwise separable convolutional layer.

    This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.

    Parameters
    ------------
    n_filter : int
        The dimensionality of the output space (i.e. the number of filters in the convolution).
    filter_size : int
        Specifying the spatial dimensions of the filters. Can be a single integer to specify the same value for all spatial dimensions.
    strides : int
        Specifying the stride of the convolution. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding : str
        One of "valid" or "same" (case-insensitive).
    data_format : str
        One of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
    dilation_rate : int
        Specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier.
    depthwise_init : initializer
        for the depthwise convolution kernel.
    pointwise_init : initializer
        For the pointwise convolution kernel.
    b_init : initializer
        For the bias vector. If None, ignore bias in the pointwise part only.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            n_filter=100,
            filter_size=3,
            strides=1,
            act=None,
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            depth_multiplier=1,
            depthwise_init=None,
            pointwise_init=None,
            b_init=tl.initializers.constant(value=0.0),
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # W_init=tf.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            in_channels=None,
            name='seperable1d',
    ):
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args
        self.in_channels = in_channels
        self.name = name

        logging.info(
            "SeparableConv1d  %s: n_filter: %d filter_size: %s strides: %s depth_multiplier: %d act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), depth_multiplier,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
             ', stride={strides}, padding={padding}')
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
        self.layer = tf.keras.layers.SeparableConv1D(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            depth_multiplier=self.depth_multiplier,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            depthwise_initializer=self.depthwise_init,
            pointwise_initializer=self.pointwise_init,
            bias_initializer=self.b_init,
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # bias_constraint=None,
            trainable=True,
            name=self.name
        )
        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]

        # _out = self.layer(np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        _out = self.layer(tf.convert_to_tensor(np.random.uniform(size=list(inputs_shape)), dtype=np.float))  # initialize weights
        outputs_shape = _out.shape
        # self._add_weights(self.layer.weights)
        self._weights = self.layer.weights

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs

    # new_variables = nn.weights
    # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
    # new_variables = get_collection_trainable(self.name)
    #
    # self._add_layers(self.outputs)
    # self._add_params(new_variables)


class SeparableConv2d(Layer):
    """The :class:`SeparableConv2d` class is a 2D depthwise separable convolutional layer.

    This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.
    While :class:`DepthwiseConv2d` performs depthwise convolution only, which allow us to add batch normalization between depthwise and pointwise convolution.

    Parameters
    ------------
    n_filter : int
        The dimensionality of the output space (i.e. the number of filters in the convolution).
    filter_size : tuple/list of 2 int
        Specifying the spatial dimensions of the filters. Can be a single integer to specify the same value for all spatial dimensions.
    strides : tuple/list of 2 int
        Specifying the strides of the convolution. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding : str
        One of "valid" or "same" (case-insensitive).
    data_format : str
        One of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
    dilation_rate : integer or tuple/list of 2 int
        Specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier.
    depthwise_init : initializer
        for the depthwise convolution kernel.
    pointwise_init : initializer
        For the pointwise convolution kernel.
    b_init : initializer
        For the bias vector. If None, ignore bias in the pointwise part only.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            n_filter=100,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            depthwise_init=None,
            pointwise_init=None,
            b_init=tl.initializers.constant(value=0.0),
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # W_init=tf.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            in_channels=None,
            name='seperable2d',
    ):
        # if W_init_args is None:
        #     W_init_args = {}
        # if b_init_args is None:
        #     b_init_args = {}

        # super(SeparableConv2d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args
        self.in_channels = in_channels
        self.name = name

        logging.info(
            "SeparableConv2d  %s: n_filter: %d filter_size: %s filter_size: %s depth_multiplier: %d act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), depth_multiplier,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
             ', stride={strides}, padding={padding}')
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
        self.layer = tf.keras.layers.SeparableConv2D(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            depth_multiplier=self.depth_multiplier,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            depthwise_initializer=self.depthwise_init,
            pointwise_initializer=self.pointwise_init,
            bias_initializer=self.b_init,
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # bias_constraint=None,
            trainable=True,
            name=self.name
        )
        if self.data_format == "channels_first":
            self.in_channels = inputs_shape[1]
        else:
            self.in_channels = inputs_shape[-1]
        # _out = self.layer(np.random.uniform([1] + list(inputs_shape)))  # initialize weights
        _out = self.layer(tf.convert_to_tensor(np.random.uniform(size=list(inputs_shape)), dtype=np.float))    # initialize weights
        outputs_shape = _out.shape
        #self._add_weights(self.layer.weights)
        self._weights = self.layer.weights

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs

    # new_variables = nn.weights
    # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
    # new_variables = get_collection_trainable(self.name)
    #
    # self._add_layers(self.outputs)
    # self._add_params(new_variables)
