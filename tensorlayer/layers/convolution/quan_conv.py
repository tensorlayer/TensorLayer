#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import quantize_active_overflow
from tensorlayer.layers.utils import quantize_weight_overflow

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['QuantizedConv2d']


class QuantizedConv2d(Layer):
    """The :class:`QuantizedConv2dWithBN` class is a quantized convolutional layer with BN, which weights are 'bitW'
    bits and the output of the previous layer are 'bitA' bits while inferencing.
    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    gemmlowp_at_inference : boolean
        If True, use gemmlowp instead of ``tf.matmul`` (gemm) for inference. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.QuantizedConv2d(net, 32, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name='qconv2d')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, name='bn1')
    ...
    >>> net = tl.layers.QuantizedConv2d(net, 64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name='qconv2d')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, name='bn2')

    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        bitW=8,
        bitA=8,
        data_format="NHWC",
        use_cudnn_on_gpu=True,
        gemmlowp_at_inference=False,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='quantized_conv2d',
    ):

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")

        if data_format not in ["NHWC", "NCHW"]:
            raise ValueError("`data_format` value is not valid, should be either: 'NHWC' or 'NCHW'")

        # TODO: Implement GEMM
        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.bitW = bitW
        self.bitA = bitA
        self.data_format = data_format
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.gemmlowp_at_inference = gemmlowp_at_inference
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(QuantizedConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %s" % self.n_filter)
        except AttributeError:
            pass

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

        try:
            input_channels = int(self._temp_data['inputs'].get_shape()[-1])
        except TypeError:  # if input_channels is ?, it happens when using Spatial Transformer Net
            input_channels = 1
            logging.warning("[warnings] unknown input channels, set to 1")

        w_shape = (self.filter_size[0], self.filter_size[1], input_channels, self.n_filter)
        strides = (1, self.strides[0], self.strides[1], 1)

        with tf.variable_scope(self.name):

            quantized_inputs = quantize_active_overflow(self._temp_data['inputs'], self.bitA)  # Do not remove

            weight_matrix = self._get_tf_variable(
                name='W_conv2d',
                shape=w_shape,
                dtype=quantized_inputs.dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            weight_matrix = quantize_weight_overflow(weight_matrix, self.bitW)

            self._temp_data['outputs'] = tf.nn.conv2d(
                quantized_inputs,
                weight_matrix,
                strides=strides,
                padding=self.padding,
                use_cudnn_on_gpu=self.use_cudnn_on_gpu,
                data_format=self.data_format
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_conv2d',
                    shape=(w_shape[-1], ),
                    dtype=quantized_inputs.dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
