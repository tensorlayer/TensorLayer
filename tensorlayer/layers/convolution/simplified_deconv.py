#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    # 'DeConv1d'  # TODO: Shall be implemented
    'DeConv2d',
    'DeConv3d',
]


class DeConv2d(Layer):
    """Simplified version of :class:`DeConv2dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    out_size : tuple of int
        Require if TF version < 1.3, (height, width) of output.
    strides : tuple of int
        The stride step (height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    batch_size : int or None
        Require if TF < 1.3, int or None.
        If None, try to find the `batch_size` from the first dim of net.outputs (you should define the `batch_size` in the input placeholder).
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (For TF < 1.3).
    b_init_args : dictionary
        The arguments for the bias vector initializer (For TF < 1.3).
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', n_out_channel='n_filter', end_support_version=1.9
    )  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            out_size=(30, 30),  # remove
            strides=(2, 2),
            padding='SAME',
            batch_size=None,  # remove
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            name='decnn2d'
    ):
        super(DeConv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DeConv2d %s: n_filters: %s strides: %s pad: %s act: %s" % (
                self.name, str(n_filter), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2, DeConv2d and DeConv2dLayer are different.")

        conv2d_transpose = tf.layers.Conv2DTranspose(
            filters=n_filter, kernel_size=filter_size, strides=strides, padding=padding, activation=self.act,
            kernel_initializer=W_init, bias_initializer=b_init, name=name
        )

        self.outputs = conv2d_transpose(self.inputs)
        # new_variables = conv2d_transpose.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)

        self._add_layers(self.outputs)
        self._add_params(new_variables)


class DeConv3d(Layer):
    """Simplified version of The :class:`DeConv3dLayer`, see `tf.contrib.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (depth, height, width).
    stride : tuple of int
        The stride step (depth, height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (For TF < 1.3).
    b_init_args : dictionary
        The arguments for the bias vector initializer (For TF < 1.3).
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='SAME',
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            name='decnn3d'
    ):
        super(DeConv3d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DeConv3d %s: n_filters: %s strides: %s pad: %s act: %s" % (
                self.name, str(n_filter), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        # with tf.variable_scope(name) as vs:
        nn = tf.layers.Conv3DTranspose(
            filters=n_filter, kernel_size=filter_size, strides=strides, padding=padding, activation=self.act,
            kernel_initializer=W_init, bias_initializer=b_init, name=name
        )

        self.outputs = nn(self.inputs)
        # new_variables = nn.weights  # tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)

        self._add_layers(self.outputs)
        self._add_params(new_variables)
