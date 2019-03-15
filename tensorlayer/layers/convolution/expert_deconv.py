#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

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
    act : activation function or None
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (height, width, output_channels, in_channels).
        The filter's ``in_channels`` dimension must match that of value.
    outputs_shape : tuple of int
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
    name : None or str
        A unique layer name.

    Notes
    -----
    - We recommend to use `DeConv2d` with TensorFlow version higher than 1.3.
    - shape = [h, w, the number of output channels of this layer, the number of output channel of the previous layer].
    - outputs_shape = [batch_size, any, any, the number of output channels of this layer].
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
    ...                            outputs_shape=(batch_size, 8, 8, 256),
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
    ...         shape=(3,3,512,1024), strides=(1,2,2,1), outputs_shape=(batch_size,64,64,512),
    ...         padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')

    """

    def __init__(
            self,
            act=None,
            shape=(3, 3, 128, 256),
            outputs_shape=(1, 256, 256, 128),
            strides=(1, 2, 2, 1),
            padding='SAME',
            dilation_rate=[1, 1],
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            # W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            # b_init=tf.compat.v1.initializers.constant(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            name=None,  #'decnn2d_layer',
    ):
        # super(DeConv2dLayer, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.act = act
        self.shape = shape
        self.outputs_shape = outputs_shape
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        logging.info(
            "DeConv2dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(outputs_shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={pre_channel}, out_channels={n_filter}, kernel_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.dilation_rate != [1,] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, n_filter=self.shape[-2], filter_size=(self.shape[0], self.shape[1]), pre_channel=self.shape[-1], **self.__dict__)

    def build(self, inputs):
        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\kernel', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.shape[-2]), init=self.b_init)
        # if self.b_init:
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\bias', shape=(self.shape[-2]), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
        #         **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):
        outputs = tf.nn.conv2d_transpose(
            input=inputs,
            filters=self.W,
            output_shape=self.outputs_shape,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            name=self.name,
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs


class DeConv3dLayer(Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`__.

    Parameters
    ----------
    act : activation function or None
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (depth, height, width, output_channels, in_channels).
        The filter's in_channels dimension must match that of value.
    outputs_shape : tuple of int
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
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            act=None,
            shape=(2, 2, 2, 128, 256),
            outputs_shape=(1, 12, 32, 32, 128),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            dilation_rate=[1, 1, 1],
            W_init=tl.initializers.truncated_normal(stddev=0.02),
            b_init=tl.initializers.constant(value=0.0),
            # W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            # b_init=tf.compat.v1.initializers.constant(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            name=None,  #'decnn3d_layer',
    ):
        # super(DeConv3dLayer, self
        # ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.act = act
        self.shape = shape
        self.outputs_shape = outputs_shape
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        logging.info(
            "DeConv3dLayer %s: shape: %s out_shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(outputs_shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(in_channels={pre_channel}, out_channels={n_filter}, kernel_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.dilation_rate != [1,] * len(self.dilation_rate):
            s += ', dilation={dilation_rate}'
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, n_filter=self.shape[-2], filter_size=(self.shape[0], self.shape[1], self.shape[2]), pre_channel=self.shape[-1], **self.__dict__)

    def build(self, inputs):
        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\kernel', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.shape[-2]), init=self.b_init)
        # if self.b_init:
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\kernel', shape=(self.shape[-2]), initializer=self.b_init,
        #         dtype=LayersConfig.tf_dtype, **self.b_init_args
        #     )

    def forward(self, inputs):
        outputs = tf.nn.conv3d_transpose(
            input=inputs,
            filters=self.W,
            output_shape=self.outputs_shape,
            strides=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate
        )
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
