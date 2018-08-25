#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['MaxPool1d', 'MaxPool2d', 'MaxPool3d']


class MaxPool1d(Layer):
    """Max pooling for 1D signal [batch, length, channel]. Wrapper for `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self, prev_layer, filter_size=3, strides=2, padding='valid', data_format='channels_last', name='maxpool1d'
    ):
        super(MaxPool1d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MaxPool1d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.max_pooling1d(
                self._temp_data['inputs'], filter_size, strides, padding=padding, data_format=data_format, name=None
            )

            self._local_weights = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self._temp_data['outputs'])
        self._add_params(self._local_weights)


class MaxPool2d(Layer):
    """Max pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(self, prev_layer, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool2d'):
        if strides is None:
            strides = filter_size

        super(MaxPool2d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MaxPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.max_pooling2d(
                self._temp_data['inputs'],
                filter_size,
                strides,
                padding=padding,
                data_format='channels_last',
                name=None
            )

            self._local_weights = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self._temp_data['outputs'])
        self._add_params(self._local_weights)


class MaxPool3d(Layer):
    """Max pooling for 3D volume [batch, depth, height, width, channel]. Wrapper for `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d>`__ .

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 3-D layer with a output rank as 5.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer,
        filter_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding='valid',
        data_format='channels_last',
        name='maxpool3d'
    ):
        super(MaxPool3d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MaxPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.max_pooling3d(
                self._temp_data['inputs'], filter_size, strides, padding=padding, data_format=data_format, name=None
            )

            self._local_weights = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self._temp_data['outputs'])
        self._add_params(self._local_weights)
