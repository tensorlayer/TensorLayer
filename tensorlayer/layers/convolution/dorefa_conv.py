#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import cabs
from tensorlayer.layers.utils import quantize_active
from tensorlayer.layers.utils import quantize_weight

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['DorefaConv2d']


class DorefaConv2d(Layer):
    """The :class:`DorefaConv2d` class is a 2D quantized convolutional layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

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
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
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
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            bitW=1,
            bitA=3,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format=None,
            use_gemm=False,
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            name=None,  #'dorefa_cnn2d',
    ):
        # super(DorefaConv2d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.bitW = bitW
        self.bitA = bitA
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.act = act
        self.padding = padding
        self.data_format = data_format
        self.use_gemm = use_gemm
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        logging.info(
            "DorefaConv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, input_shape):

        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        if len(self.strides) != 2:
            raise ValueError("len(strides) should be 2.")

        try:
            self.pre_channel = input_shape[-1]
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            self.pre_channel = 1
            logging.warning("[warnings] unknow input channels, set to 1")

        self.shape = (self.filter_size[0], self.filter_size[1], self.pre_channel, self.n_filter)
        self.strides = (1, self.strides[0], self.strides[1], 1)

        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\kernel', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init, init_args=self.W_init_args)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init, init_args=self.b_init_args)
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\bias', shape=(self.shape[-1]), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
        #         **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):

        inputs = quantize_active(cabs(inputs), self.bitA)  # Do not remove

        W_ = quantize_weight(self.W, self.bitW)

        outputs = tf.nn.conv2d(
            inputs, W_, strides=self.strides, padding=self.padding, use_cudnn_on_gpu=self.use_cudnn_on_gpu,
            data_format=self.data_format
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        if self.act:
            outputs = self.act(outputs)

        return outputs
