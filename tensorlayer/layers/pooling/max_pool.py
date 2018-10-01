#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['MaxPool1d', 'MaxPool2d', 'MaxPool3d']


class MaxPool1d(Layer):
    """Max pooling for 1D signal.

    Parameters
    ----------
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

    def __init__(self, filter_size=3, strides=2, padding='same', data_format='channels_last', name='maxpool_1d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.name = name

        super(MaxPool1d, self).__init__()

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

    def build(self):

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.max_pooling1d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="max_pooling_1d_op"
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MaxPool2d(Layer):
    """Max pooling for 2D image.

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
        self, filter_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last', name='maxpool_2d'
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

        super(MaxPool2d, self).__init__()

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

    def build(self):

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.max_pooling2d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="maxpool_2d_op"
            )

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class MaxPool3d(Layer):
    """Max pooling for 3D volume.

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
        self, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_last', name='maxpool_3d'
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

        super(MaxPool3d, self).__init__()

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

    def build(self):

        with tf.variable_scope(self.name):

            self._temp_data['outputs'] = tf.layers.max_pooling3d(
                inputs=self._temp_data['inputs'],
                pool_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                name="maxpool_3d_op"
            )
