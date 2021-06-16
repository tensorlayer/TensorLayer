#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = [
    'DorefaConv2d',
]


class DorefaConv2d(Module):
    """The :class:`DorefaConv2d` class is a 2D quantized convolutional layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    dilation_rate : tuple of int
        Specifying the dilation rate to use for dilated convolution.
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([8, 12, 12, 32], name='input')
    >>> dorefaconv2d = tl.layers.DorefaConv2d(
    ...     n_filter=32, filter_size=(5, 5), strides=(1, 1), act=tl.ReLU, padding='SAME', name='dorefaconv2d'
    ... )(net)
    >>> print(dorefaconv2d)
    >>> output shape : (8, 12, 12, 32)

    """

    def __init__(
        self,
        bitW=1,
        bitA=3,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=None,
        padding='SAME',
        data_format="channels_last",
        dilation_rate=(1, 1),
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None  # 'dorefa_cnn2d',
    ):
        super().__init__(name, act=act)
        self.bitW = bitW
        self.bitA = bitA
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = self._dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels:
            self.build(None)
            self._built = True

        logging.info(
            "DorefaConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
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

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)
            self.bias_add = tl.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

        self.dorefaconv2d = tl.ops.DorefaConv2D(
            bitW=self.bitW, bitA=self.bitA, strides=self._strides, padding=self.padding, data_format=self.data_format,
            dilations=self._dilation_rate, out_channel=self.n_filter, k_size=self.filter_size,
            in_channel=self.in_channels
        )

    def forward(self, inputs):

        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.dorefaconv2d(inputs, self.W)

        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs
