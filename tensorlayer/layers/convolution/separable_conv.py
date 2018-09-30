#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'SeparableConv1d',
    'SeparableConv2d',
]


class SeparableConv1d(Layer):
    """The :class:`SeparableConv1d` class is a 1D depthwise separable convolutional layer, see `tf.layers.separable_conv1d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv1d>`__.

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
    name : a str
        A unique layer name.

    """

    def __init__(
        self,
        n_filter=100,
        filter_size=3,
        strides=1,
        padding='valid',
        data_format='channels_last',
        dilation_rate=1,
        depth_multiplier=1,
        depthwise_init=None,
        pointwise_init=None,
        b_init=tf.zeros_initializer(),
        act=None,
        name='separable_conv1d',
    ):

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("`data_format` value is not valid, should be either: 'channels_last' or 'channels_first'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(SeparableConv1d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %d" % self.filter_size)
        except AttributeError:
            pass

        try:
            additional_str.append("strides: %s" % self.strides)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("dilation_rate: %s" % self.dilation_rate)
        except AttributeError:
            pass

        try:
            additional_str.append("depth_multiplier: %s" % self.depth_multiplier)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        is_name_reuse = tf.get_variable_scope().reuse

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.separable_conv1d(
                inputs=self._temp_data['inputs'],
                filters=self.n_filter,
                kernel_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                depth_multiplier=self.depth_multiplier,
                activation=None,
                use_bias=(True if self.b_init is not None else False),
                depthwise_initializer=self.depthwise_init,
                pointwise_initializer=self.pointwise_init,
                bias_initializer=self.b_init,
                trainable=self._temp_data['is_train'],
                reuse=is_name_reuse,
                name=None
            )

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class SeparableConv2d(Layer):
    """The :class:`SeparableConv2d` class is a 2D depthwise separable convolutional layer, see `tf.layers.separable_conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d>`__.

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
    name : a str
        A unique layer name.

    """

    def __init__(
        self,
        n_filter=100,
        filter_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        depth_multiplier=1,
        depthwise_init=None,
        pointwise_init=None,
        b_init=tf.zeros_initializer(),
        act=None,
        name='separable_conv2d',
    ):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("`data_format` value is not valid, should be either: 'channels_last' or 'channels_first'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(SeparableConv2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("dilation_rate: %s" % str(self.dilation_rate))
        except AttributeError:
            pass

        try:
            additional_str.append("depth_multiplier: %s" % self.depth_multiplier)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        is_name_reuse = tf.get_variable_scope().reuse

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.separable_conv2d(
                inputs=self._temp_data['inputs'],
                filters=self.n_filter,
                kernel_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                depth_multiplier=self.depth_multiplier,
                use_bias=(True if self.b_init is not None else False),
                depthwise_initializer=self.depthwise_init,
                pointwise_initializer=self.pointwise_init,
                bias_initializer=self.b_init,
                trainable=self._temp_data['is_train'],
                reuse=is_name_reuse,
                activation=None,
                name=None
            )

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
