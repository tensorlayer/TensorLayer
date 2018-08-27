#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['MeanPool1d', 'MeanPool2d', 'MeanPool3d']


class MeanPool1d(Layer):
    """Mean pooling for 1D signal.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    """

    def __init__(
        self,  #prev_layer,
        filter_size=3,
        strides=2,
        padding='valid',
        data_format='channels_last',
        name='meanpool1d'
    ):
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.name = name
        super(MeanPool1d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("filter_size: %d" % self.filter_size)
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % self.stride)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.average_pooling1d(
                prev_layer.outputs,
                self.filter_size,
                self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name=None
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MeanPool2d(Layer):
    """Mean pooling for 2D image.

    Parameters
    -----------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width]
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    """

    # @deprecated_alias(
    #     layer='prev_layer', end_support_version="2.0.0"
    # )  # TODO: remove this line before releasing TL 2.0.0
    # @deprecated_args(
    #     end_support_version="2.1.0",
    #     instructions="`prev_layer` is deprecated, use the functional API instead",
    #     deprecated_args=("prev_layer", ),
    # )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,  # prev_layer,
        filter_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        data_format='channels_last',
        name='meanpool2d'
    ):

        if strides is None:
            strides = filter_size
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.name = name
        super(MeanPool2d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % self.stride)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
        """
        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.average_pooling2d(
                self._temp_data['inputs'],
                self.filter_size,
                self.strides,
                padding=self.padding,
                data_format='channels_last',
                name=None
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MeanPool3d(Layer):
    """Mean pooling for 3D volume.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    # Returns
    # -------
    # :class:`Layer`
    #     A mean pooling 3-D layer with a output rank as 5.

    """

    # @deprecated_alias(
    #     layer='prev_layer', end_support_version="2.0.0"
    # )  # TODO: remove this line before releasing TL 2.0.0
    # @deprecated_args(
    #     end_support_version="2.1.0",
    #     instructions="`prev_layer` is deprecated, use the functional API instead",
    #     deprecated_args=("prev_layer", ),
    # )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        # prev_layer,
        filter_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding='valid',
        data_format='channels_last',
        name='meanpool3d'
    ):
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.name = name
        super(MeanPool3d, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % self.stride)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.average_pooling3d(
                prev_layer.outputs,
                self.filter_size,
                self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name=None
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
