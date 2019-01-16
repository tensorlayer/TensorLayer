#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import logging

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
    # prev_layer : :class:`Layer`
    #     Previous layer.
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
    act : activation function
        The activation function of this layer.
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (For TF < 1.3).
    b_init_args : dictionary
        The arguments for the bias vector initializer (For TF < 1.3).
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            # prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            # out_size=(30, 30),  # remove
            strides=(2, 2),
            padding='SAME',
            # batch_size=None,  # remove
            act=None,
            data_format='channels_last',
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            name=None, #'decnn2d'
    ):
        # super(DeConv2d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.n_filter=n_filter
        self.filter_size=filter_size
        self.strides=strides
        self.padding=padding
        self.act=act
        self.data_format=data_format
        self.W_init=W_init
        self.b_init=b_init
        self.W_init_args=W_init_args  # TODO: Remove when TF <1.3 not supported
        self.b_init_args=b_init_args  # TODO: Remove when TF <1.3 not supported

        logging.info(
            "DeConv2d %s: n_filters: %s strides: %s pad: %s act: %s" % (
                self.name, str(n_filter), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2, DeConv2d and DeConv2dLayer are different.")

    def build(self, inputs_shape):
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            name=self.name,
        )

        _out = self.layer(np.random.uniform([1]+list(inputs_shape))) # initialize weights
        outputs_shape = _out.shape
        self._add_weights(self.layer.weights)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs
        # self.outputs = conv2d_transpose(self.inputs)
        # # new_variables = conv2d_transpose.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        # new_variables = get_collection_trainable(self.name)
        #
        # self._add_layers(self.outputs)
        # self._add_params(new_variables)


class DeConv3d(Layer):
    """Simplified version of The :class:`DeConv3dLayer`, see `tf.contrib.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d_transpose>`__.

    Parameters
    ----------
    # prev_layer : :class:`Layer`
    #     Previous layer.
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
    data_format : str
        "channels_last" (NDHWC, default) or "channels_first" (NCDHW).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (For TF < 1.3).
    b_init_args : dictionary
        The arguments for the bias vector initializer (For TF < 1.3).
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            # prev_layer,
            n_filter=32,
            filter_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='SAME',
            act=None,
            data_format='channels_last',
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            name=None, #'decnn3d'
    ):
        # super(DeConv3d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.n_filter=n_filter
        self.filter_size=filter_size
        self.strides=strides
        self.padding=padding
        self.act=act
        self.data_format=data_format
        self.W_init=W_init
        self.b_init=b_init
        self.W_init_args=W_init_args  # TODO: Remove when TF <1.3 not supported
        self.b_init_args=b_init_args  # TODO: Remove when TF <1.3 not supported

        logging.info(
            "DeConv3d %s: n_filters: %s strides: %s pad: %s act: %s" % (
                self.name, str(n_filter), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs_shape):
        # with tf.variable_scope(name) as vs:
        self.layer = tf.keras.layers.Conv3DTranspose(
            filters=self.n_filter,
            kernel_size=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.act,
            use_bias=(True if self.b_init is not None else False),
            data_format=self.data_format,
            kernel_initializer=self.W_init,
            bias_initializer=self.b_init,
            name=self.name,
        )

        _out = self.layer(np.random.uniform([1]+list(inputs_shape))) # initialize weights
        outputs_shape = _out.shape
        self._add_weights(self.layer.weights)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        return outputs
        # self.outputs = nn(self.inputs)
        # # new_variables = nn.weights  # tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        # new_variables = get_collection_trainable(self.name)
        #
        # self._add_layers(self.outputs)
        # self._add_params(new_variables)
