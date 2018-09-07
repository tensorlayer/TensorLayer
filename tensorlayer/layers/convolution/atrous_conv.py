#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.convolution.expert_conv import Conv1dLayer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'AtrousDeConv2dLayer',
]


@deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
def atrous_conv1d(
        prev_layer,
        n_filter=32,
        filter_size=2,
        stride=1,
        dilation=1,
        act=None,
        padding='SAME',
        data_format='NWC',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        name='atrous_1d',
):
    """Simplified version of :class:`AtrousConv1dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : int
        The filter size.
    stride : tuple of int
        The strides: (height, width).
    dilation : int
        The filter dilation size.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A :class:`AtrousConv1dLayer` object

    """
    return Conv1dLayer(
        prev_layer=prev_layer,
        act=act,
        shape=(filter_size, int(prev_layer.outputs.get_shape()[-1]), n_filter),
        stride=stride,
        padding=padding,
        dilation_rate=dilation,
        data_format=data_format,
        W_init=W_init,
        b_init=b_init,
        W_init_args=W_init_args,
        b_init_args=b_init_args,
        name=name,
    )


class AtrousConv2dLayer(Layer):
    """The :class:`AtrousConv2dLayer` class is 2D atrous convolution (a.k.a. convolution with holes or dilated
    convolution) 2D layer, see `tf.nn.atrous_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#atrous_conv2d>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with a 4D output tensor in the shape of (batch, height, width, channels).
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size: (height, width).
    rate : int
        The stride that we sample input values in the height and width dimensions.
        This equals the rate that we up-sample the filters by inserting zeros across the height and width dimensions.
        In the literature, this parameter is sometimes mentioned as input stride or dilation.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=(3, 3), rate=2, act=None, padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
            W_init_args=None, b_init_args=None, name='atrous_2d'
    ):

        super(AtrousConv2dLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "AtrousConv2dLayer %s: n_filter: %d filter_size: %s rate: %d pad: %s act: %s" % (
                self.name, n_filter, filter_size, rate, padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        with tf.variable_scope(name):
            shape = [filter_size[0], filter_size[1], int(self.inputs.get_shape()[-1]), n_filter]

            W = tf.get_variable(
                name='W_atrous_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            self.outputs = tf.nn.atrous_conv2d(self.inputs, filters=W, rate=rate, padding=padding)

            if b_init:
                b = tf.get_variable(
                    name='b_atrous_conv2d', shape=(n_filter), initializer=b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:
            self._add_params(W)


class AtrousDeConv2dLayer(Layer):
    """The :class:`AtrousDeConv2dLayer` class is 2D atrous convolution transpose, see `tf.nn.atrous_conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#atrous_conv2d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with a 4D output tensor in the shape of (batch, height, width, channels).
    shape : tuple of int
        The shape of the filters: (filter_height, filter_width, out_channels, in_channels).
    output_shape : tuple of int
        Output shape of the deconvolution.
    rate : int
        The stride that we sample input values in the height and width dimensions.
        This equals the rate that we up-sample the filters by inserting zeros across the height and width dimensions.
        In the literature, this parameter is sometimes mentioned as input stride or dilation.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, shape=(3, 3, 128, 256), output_shape=(1, 64, 64, 128), rate=2, act=None, padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
            W_init_args=None, b_init_args=None, name='atrous_2d_transpose'
    ):

        super(AtrousDeConv2dLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "AtrousDeConv2dLayer %s: shape: %s output_shape: %s rate: %d pad: %s act: %s" % (
                self.name, shape, output_shape, rate, padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_atrous_conv2d_transpose', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype,
                **self.W_init_args
            )

            self.outputs = tf.nn.atrous_conv2d_transpose(
                self.inputs, filters=W, output_shape=output_shape, rate=rate, padding=padding
            )

            if b_init:
                b = tf.get_variable(
                    name='b_atrous_conv2d_transpose', shape=(shape[-2]), initializer=b_init,
                    dtype=LayersConfig.tf_dtype, **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:

            self._add_params(W)


# Alias
AtrousConv1dLayer = atrous_conv1d
