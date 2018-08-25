#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import compute_alpha
from tensorlayer.layers.utils import ternary_operation

from tensorlayer import logging

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['TernaryConv2d']


class TernaryConv2d(Layer):
    """
    The :class:`TernaryConv2d` class is a 2D binary CNN layer, which weights are either -1 or 1 or 0 while inference.

    Note that, the bias vector would not be tenarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
    >>> net = tl.layers.TernaryConv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='ternary_conv2d_1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=True, name='bn1')
    ...
    >>> net = tl.layers.SignLayer(net)
    >>> net = tl.layers.TernaryConv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='ternary_conv2d_2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=True, name='bn2')

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
        prev_layer=None,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        data_format="NHWC",
        use_cudnn_on_gpu=True,
        gemmlowp_at_inference=False,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='ternary_conv2d',
    ):

        if data_format not in ["NHWC", "NCHW"]:
            raise ValueError("`data_format` value is not valid, should be either: 'NHWC' or 'NCHW'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")

        # TODO: Implement GEMM
        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        self.prev_layer = prev_layer
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.gemmlowp_at_inference = gemmlowp_at_inference
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(TernaryConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        try:
            input_channels = int(self._temp_data['inputs'].get_shape()[-1])

        except TypeError:  # if input_channels is ?, it happens when using Spatial Transformer Net
            input_channels = 1
            logging.warning("unknow input channels, set to 1")

        shape = (self.filter_size[0], self.filter_size[1], input_channels, self.n_filter)
        strides = (1, self.strides[0], self.strides[1], 1)

        with tf.variable_scope(self.name):

            weight_matrix = self._get_tf_variable(
                name='W_conv2d',
                shape=shape,
                initializer=self.W_init,
                dtype=self._temp_data['inputs'].dtype,
                **self.W_init_args
            )

            alpha = compute_alpha(weight_matrix)

            weight_matrix = ternary_operation(weight_matrix)
            weight_matrix = tf.multiply(alpha, weight_matrix)

            self._temp_data['outputs'] = tf.nn.conv2d(
                self._temp_data['inputs'],
                weight_matrix,
                strides=strides,
                padding=self.padding,
                use_cudnn_on_gpu=self.use_cudnn_on_gpu,
                data_format=self.data_format
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_conv2d',
                    shape=(shape[-1]),
                    initializer=self.b_init,
                    dtype=self._temp_data['inputs'].dtype,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
