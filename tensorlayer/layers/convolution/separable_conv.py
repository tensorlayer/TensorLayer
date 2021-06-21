#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module
from tensorlayer.backend import BACKEND

__all__ = [
    'SeparableConv1d',
    'SeparableConv2d',
]


class SeparableConv1d(Module):
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
    act : activation function
        The activation function of this layer.
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

    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([8, 50, 64], name='input')
    >>> separableconv1d = tl.layers.SeparableConv1d(n_filter=32, filter_size=3, strides=2, padding='SAME', act=tl.ReLU, name='separable_1d')(net)
    >>> print(separableconv1d)
    >>> output shape : (8, 25, 32)

    """

    def __init__(
        self, n_filter=32, filter_size=1, stride=1, act=None, padding="SAME", data_format="channels_last",
        dilation_rate=1, depth_multiplier=1, depthwise_init=tl.initializers.truncated_normal(stddev=0.02),
        pointwise_init=tl.initializers.truncated_normal(stddev=0.02), b_init=tl.initializers.constant(value=0.0),
        in_channels=None, name=None
    ):
        super(SeparableConv1d, self).__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "SeparableConv1d  %s: n_filter: %d filter_size: %s strides: %s depth_multiplier: %d act: %s" % (
                self.name, n_filter, str(filter_size), str(stride), depth_multiplier,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', stride={strides}, padding={padding}'
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

        if BACKEND == 'tensorflow':
            self.depthwise_filter_shape = (self.filter_size, self.in_channels, self.depth_multiplier)
        elif BACKEND == 'mindspore':
            self.depthwise_filter_shape = (self.filter_size, 1, self.depth_multiplier * self.in_channels)

        self.pointwise_filter_shape = (1, self.depth_multiplier * self.in_channels, self.n_filter)

        self.depthwise_W = self._get_weights(
            'depthwise_filters', shape=self.depthwise_filter_shape, init=self.depthwise_init
        )
        self.pointwise_W = self._get_weights(
            'pointwise_filters', shape=self.pointwise_filter_shape, init=self.pointwise_init
        )

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)
            self.bias_add = tl.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.act_init_flag = False
        if self.act:
            self.activate = self.act
            self.act_init_flag = True

        self.separable_conv1d = tl.ops.SeparableConv1D(
            stride=self.stride, padding=self.padding, data_format=self.data_format, dilations=self.dilation_rate,
            out_channel=self.n_filter, k_size=self.filter_size, in_channel=self.in_channels,
            depth_multiplier=self.depth_multiplier
        )

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.separable_conv1d(inputs, self.depthwise_W, self.pointwise_W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs


class SeparableConv2d(Module):
    """The :class:`SeparableConv2d` class is a 2D depthwise separable convolutional layer.
        This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.

        Parameters
        ------------
        n_filter : int
            The dimensionality of the output space (i.e. the number of filters in the convolution).
        filter_size : tuple of int
            Specifying the spatial dimensions of the filters. Can be a single integer to specify the same value for all spatial dimensions.
        strides : tuple of int
            Specifying the stride of the convolution. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
        act : activation function
            The activation function of this layer.
        padding : str
            One of "valid" or "same" (case-insensitive).
        data_format : str
            One of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
        dilation_rate : tuple of int
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

        Examples
        --------
        With TensorLayer
        >>> net = tl.layers.Input([8, 50, 50, 64], name='input')
        >>> separableconv2d = tl.layers.SeparableConv2d(n_filter=32, filter_size=3, strides=2, depth_multiplier = 3 , padding='SAME', act=tl.ReLU, name='separable_2d')(net)
        >>> print(separableconv2d)
        >>> output shape : (8, 24, 24, 32)

        """

    def __init__(
        self, n_filter=32, filter_size=(1, 1), strides=(1, 1), act=None, padding="VALID", data_format="channels_last",
        dilation_rate=(1, 1), depth_multiplier=1, depthwise_init=tl.initializers.truncated_normal(stddev=0.02),
        pointwise_init=tl.initializers.truncated_normal(stddev=0.02), b_init=tl.initializers.constant(value=0.0),
        in_channels=None, name=None
    ):
        super(SeparableConv2d, self).__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self._strides = self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self._dilation_rate = self.dilation_rate = dilation_rate
        self.depth_multiplier = depth_multiplier
        self.depthwise_init = depthwise_init
        self.pointwise_init = pointwise_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "SeparableConv2d  %s: n_filter: %d filter_size: %s strides: %s depth_multiplier: %d act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), depth_multiplier,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', stride={strides }, padding={padding}'
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

        if BACKEND == 'tensorflow':
            self.depthwise_filter_shape = (
                self.filter_size[0], self.filter_size[1], self.in_channels, self.depth_multiplier
            )
            self.pointwise_filter_shape = (1, 1, self.depth_multiplier * self.in_channels, self.n_filter)

        elif BACKEND == 'mindspore':
            self.depthwise_filter_shape = (
                self.filter_size[0], self.filter_size[1], 1, self.depth_multiplier * self.in_channels
            )
            self.pointwise_filter_shape = (1, 1, self.depth_multiplier * self.in_channels, self.n_filter)

        self.depthwise_W = self._get_weights(
            'depthwise_filters', shape=self.depthwise_filter_shape, init=self.depthwise_init
        )

        self.pointwise_W = self._get_weights(
            'pointwise_filters', shape=self.pointwise_filter_shape, init=self.pointwise_init
        )

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)
            self.bias_add = tl.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.separable_conv2d = tl.ops.SeparableConv2D(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.n_filter, k_size=self.filter_size, in_channel=self.in_channels,
            depth_multiplier=self.depth_multiplier
        )

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.separable_conv2d(inputs, self.depthwise_W, self.pointwise_W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs
