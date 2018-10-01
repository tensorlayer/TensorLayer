#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args
from tensorlayer.decorators import private_method

__all__ = [
    'GroupConv2d',
]


class GroupConv2d(Layer):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

    Parameters
    --------------
    n_filter : int
        The number of filters.
    filter_size : tuple of ints
        The filter size.
    stride : tuple of ints
        The stride step.
    n_group : int
        The number of groups.
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

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(2, 2),
        n_group=2,
        padding='SAME',
        data_format="NHWC",
        use_cudnn_on_gpu=True,
        gemmlowp_at_inference=False,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='groupconv2d',
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
        self.n_group = n_group
        self.padding = padding
        self.data_format = data_format
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.gemmlowp_at_inference = gemmlowp_at_inference
        self.padding = padding
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(GroupConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

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
            additional_str.append("n_group: %s" % self.n_group)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        input_channels = int(self._temp_data['inputs'].get_shape()[-1])

        if input_channels % self.n_group != 0:
            raise ValueError("The number of input channels must be evenly divisible by `n_group`")

        if self.n_filter % self.n_group != 0:
            raise ValueError("`n_filter` must be evenly divisible by `n_group`")

        with tf.variable_scope(self.name):

            We = self._get_tf_variable(
                name='W',
                shape=[self.filter_size[0], self.filter_size[1], input_channels / self.n_group, self.n_filter],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            def _exec_conv2d(inputs, n_filters):
                return tf.nn.conv2d(
                    input=inputs,
                    filter=n_filters,
                    strides=[1, self.strides[0], self.strides[1], 1],
                    padding=self.padding,
                    data_format=self.data_format,
                    use_cudnn_on_gpu=self.use_cudnn_on_gpu
                )

            if self.n_group == 1:
                self._temp_data['outputs'] = _exec_conv2d(self._temp_data['inputs'], We)

            else:
                input_groups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self._temp_data['inputs'])
                weights_groups = tf.split(axis=3, num_or_size_splits=self.n_group, value=We)

                conv_groups = [
                    _exec_conv2d(inputs, n_filters) for inputs, n_filters in zip(input_groups, weights_groups)
                ]

                self._temp_data['outputs'] = tf.concat(axis=3, values=conv_groups)

            if self.b_init:
                b = self._get_tf_variable(
                    name='b',
                    shape=self.n_filter,
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
