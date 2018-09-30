#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['MeanPool1d', 'MeanPool2d', 'MeanPool3d']


class MeanPool1d(Layer):
    """Mean pooling for 1D signal.

    Parameters
    ------------
    filter_size : int
        Pooling window size.
    strides : int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    """

    def __init__(self, filter_size=3, strides=2, padding='same', data_format='channels_last', name='meanpool_1d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

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
            additional_str.append("strides: %s" % self.strides)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.average_pooling1d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="avg_pooling_1d_op"
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MeanPool2d(Layer):
    """Mean pooling for 2D image.

    Parameters
    -----------
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

    def __init__(
        self, filter_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='meanpool_2d'
    ):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

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
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.average_pooling2d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="avg_pooling_2d_op"
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MeanPool3d(Layer):
    """Mean pooling for 3D volume.

    Parameters
    ------------
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

    """

    def __init__(
        self, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_last', name='meanpool3d'
    ):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

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
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self):
        with tf.variable_scope(self.name):

            self._temp_data['outputs'] = tf.layers.average_pooling3d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="avg_pooling_3d_op"
            )
