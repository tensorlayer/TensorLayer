#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module
from tensorlayer.backend import BACKEND

__all__ = [
    'GroupConv2d',
]


class GroupConv2d(Module):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

      Parameters
      --------------
      n_filter : int
          The number of filters.
      filter_size : tuple of int
          The filter size.
      stride : tuple of int
          The stride step.
      n_group : int
          The number of groups.
      act : activation function
          The activation function of this layer.
      padding : str
          The padding algorithm type: "SAME" or "VALID".
      data_format : str
          "channels_last" (NHWC, default) or "channels_first" (NCHW).
      dilation_rate : tuple of int
          Specifying the dilation rate to use for dilated convolution.
      W_init : initializer
          The initializer for the weight matrix.
      b_init : initializer or None
          The initializer for the bias vector. If None, skip biases.
      in_channels : int
          The number of in channels.
      name : None or str
          A unique layer name.

      Examples
      ---------
      With TensorLayer
      >>> net = tl.layers.Input([8, 24, 24, 32], name='input')
      >>> groupconv2d = tl.layers.QuanConv2d(
      ...     n_filter=64, filter_size=(3, 3), strides=(2, 2), n_group=2, name='group'
      ... )(net)
      >>> print(groupconv2d)
      >>> output shape : (8, 12, 12, 64)

      """

    def __init__(
        self, n_filter=32, filter_size=(1, 1), strides=(1, 1), n_group=1, act=None, padding='SAME',
        data_format="channels_last", dilation_rate=(1, 1), W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0), in_channels=None, name=None
    ):
        super().__init__(name, act=act)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self._strides = self.strides = strides
        self.n_group = n_group
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
            "Conv2d %s: n_filter: %d filter_size: %s strides: %s n_group: %d pad: %s  act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), n_group, padding,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else "No Activation"
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', strides={strides}, n_group = {n_group}, padding={padding}'
        )
        if self.dilation_rate != (1, ) * len(self.dilation_rate):
            s += ', dilation = {dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (',', +actstr)
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

        if self.n_group < 1:
            raise ValueError(
                "The n_group must be a integer greater than or equal to 1, but we got :{}".format(self.n_group)
            )

        if self.in_channels % self.n_group != 0:
            raise ValueError(
                "The channels of input must be divisible by n_group, but we got: the channels of input"
                "is {}, the n_group is {}.".format(self.in_channels, self.n_group)
            )

        if self.n_filter % self.n_group != 0:
            raise ValueError(
                "The number of filters must be divisible by n_group, but we got: the number of filters "
                "is {}, the n_group is {}. ".format(self.n_filter, self.n_group)
            )

        # TODO channels first filter shape [out_channel, in_channel/n_group, filter_h, filter_w]
        self.filter_shape = (
            self.filter_size[0], self.filter_size[1], int(self.in_channels / self.n_group), self.n_filter
        )

        self.W = self._get_weights("filters", shape=self.filter_shape, init=self.W_init)

        self.b_init_flag = False
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter, ), init=self.b_init)
            self.bias_add = tl.ops.BiasAdd(self.data_format)
            self.b_init_flag = True

        self.group_conv2d = tl.ops.GroupConv2D(
            strides=self._strides, padding=self.padding, data_format=self.data_format, dilations=self._dilation_rate,
            out_channel=self.n_filter, k_size=(self.filter_size[0], self.filter_size[1]), groups=self.n_group
        )

        self.act_init_flag = False
        if self.act:
            self.act_init_flag = True

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        outputs = self.group_conv2d(inputs, self.W)
        if self.b_init_flag:
            outputs = self.bias_add(outputs, self.b)
        if self.act_init_flag:
            outputs = self.act(outputs)
        return outputs
