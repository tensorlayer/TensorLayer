#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'DeConv2dLayer',
    'DeConv3dLayer',
]


class DeConv2dLayer(Layer):
    """A de-convolution 2D layer.

    See `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (height, width, output_channels, in_channels).
        The filter's ``in_channels`` dimension must match that of value.
    output_shape : tuple of int
        Output shape of the deconvolution,
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for initializing the weight matrix.
    b_init_args : dictionary
        The arguments for initializing the bias vector.
    name : str
        A unique layer name.

    Notes
    -----
    - We recommend to use `DeConv2d` with TensorFlow version higher than 1.3.
    - shape = [h, w, the number of output channels of this layer, the number of output channel of the previous layer].
    - output_shape = [batch_size, any, any, the number of output channels of this layer].
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    A part of the generator in DCGAN example

    >>> batch_size = 64
    >>> inputs = tf.placeholder(tf.float32, [batch_size, 100], name='z_noise')
    >>> net_in = tl.layers.InputLayer(inputs, name='g/in')
    >>> net_h0 = tl.layers.DenseLayer(net_in, n_units = 8192,
    ...                            W_init = tf.random_normal_initializer(stddev=0.02),
    ...                            act = None, name='g/h0/lin')
    >>> print(net_h0.outputs._shape)
    (64, 8192)
    >>> net_h0 = tl.layers.ReshapeLayer(net_h0, shape=(-1, 4, 4, 512), name='g/h0/reshape')
    >>> net_h0 = tl.layers.BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train, name='g/h0/batch_norm')
    >>> print(net_h0.outputs._shape)
    (64, 4, 4, 512)
    >>> net_h1 = tl.layers.DeConv2dLayer(net_h0,
    ...                            shape=(5, 5, 256, 512),
    ...                            output_shape=(batch_size, 8, 8, 256),
    ...                            strides=(1, 2, 2, 1),
    ...                            act=None, name='g/h1/decon2d')
    >>> net_h1 = tl.layers.BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train, name='g/h1/batch_norm')
    >>> print(net_h1.outputs._shape)
    (64, 8, 8, 256)

    U-Net

    >>> ....
    >>> conv10 = tl.layers.Conv2dLayer(conv9, act=tf.nn.relu,
    ...        shape=(3,3,1024,1024), strides=(1,1,1,1), padding='SAME',
    ...        W_init=w_init, b_init=b_init, name='conv10')
    >>> print(conv10.outputs)
    (batch_size, 32, 32, 1024)
    >>> deconv1 = tl.layers.DeConv2dLayer(conv10, act=tf.nn.relu,
    ...         shape=(3,3,512,1024), strides=(1,2,2,1), output_shape=(batch_size,64,64,512),
    ...         padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=None,
            shape=(3, 3, 128, 256),
            output_shape=(1, 256, 256, 128),
            strides=(1, 2, 2, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='decnn2d_layer',
    ):
        super(DeConv2dLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DeConv2dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(output_shape), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        # logging.info("  DeConv2dLayer: Untested")
        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_deconv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            self.outputs = tf.nn.conv2d_transpose(
                self.inputs, W, output_shape=output_shape, strides=strides, padding=padding
            )

            if b_init:
                b = tf.get_variable(
                    name='b_deconv2d', shape=(shape[-2]), initializer=b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )
                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:
            self._add_params(W)


class DeConv3dLayer(Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (depth, height, width, output_channels, in_channels).
        The filter's in_channels dimension must match that of value.
    output_shape : tuple of int
        The output shape of the deconvolution.
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
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
            self,
            prev_layer,
            act=None,
            shape=(2, 2, 2, 128, 256),
            output_shape=(1, 12, 32, 32, 128),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='decnn3d_layer',
    ):
        super(DeConv3dLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DeConv3dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(output_shape), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation'
            )
        )

        with tf.variable_scope(name):

            W = tf.get_variable(
                name='W_deconv3d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            self.outputs = tf.nn.conv3d_transpose(
                self.inputs, W, output_shape=output_shape, strides=strides, padding=padding
            )

            if b_init:
                b = tf.get_variable(
                    name='b_deconv3d', shape=(shape[-2]), initializer=b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([W, b])
        else:
            self._add_params([W])
