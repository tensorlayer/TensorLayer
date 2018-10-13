#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils import compute_deconv2d_output_shape

from tensorlayer.layers.convolution.expert_conv import Conv1dLayer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'AtrousDeConv2d',
]


class AtrousDeConv2d(Layer):
    """The :class:`AtrousDeConv2d` class is 2D atrous convolution transpose, see `tf.nn.atrous_conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#atrous_conv2d_transpose>`__.

    Parameters
    ----------
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

    def __init__(
        self,
        shape=(3, 3, 128, 256),
        rate=2,
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='atrous_2d_transpose'
    ):

        padding = padding.upper()

        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.shape = shape
        self.rate = rate
        self.padding = padding
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(AtrousDeConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("shape: %s" % str(self.shape))
        except AttributeError:
            pass

        try:
            additional_str.append("rate: %d" % self.rate)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            weight_matrix = self._get_tf_variable(
                name='W_atrous_conv2d_transpose',
                shape=self.shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            out_shape = compute_deconv2d_output_shape(
                self._temp_data['inputs'],
                self.shape[0],
                self.shape[1],
                1,
                1,
                self.shape[2],
                padding=self.padding,
                data_format="NHWC"
            )

            self._temp_data['outputs'] = tf.nn.atrous_conv2d_transpose(
                self._temp_data['inputs'],
                filters=weight_matrix,
                output_shape=out_shape,
                rate=self.rate,
                padding=self.padding
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_atrous_conv2d_transpose',
                    shape=(self.shape[-2], ),
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
