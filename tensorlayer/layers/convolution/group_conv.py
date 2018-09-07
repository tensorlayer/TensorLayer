#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'GroupConv2d',
]


class GroupConv2d(Layer):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

    Parameters
    --------------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : int
        The filter size.
    stride : int
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(2, 2),
            n_group=2,
            act=None,
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # TODO: Remove when TF <1.3 not supported
            b_init_args=None,  # TODO: Remove when TF <1.3 not supported
            name='groupconv',
    ):  # Windaway

        super(GroupConv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "GroupConv2d %s: n_filter: %d size: %s strides: %s n_group: %d pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), n_group, padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strides[0], strides[1], 1], padding=padding)
        channels = int(self.inputs.get_shape()[-1])

        with tf.variable_scope(name):
            We = tf.get_variable(
                name='W', shape=[filter_size[0], filter_size[1], channels / n_group, n_filter], initializer=W_init,
                dtype=LayersConfig.tf_dtype, trainable=True, **self.W_init_args
            )

            if n_group == 1:
                self.outputs = groupConv(self.inputs, We)

            else:
                inputGroups = tf.split(axis=3, num_or_size_splits=n_group, value=self.inputs)
                weightsGroups = tf.split(axis=3, num_or_size_splits=n_group, value=We)
                convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]

                self.outputs = tf.concat(axis=3, values=convGroups)

            if b_init:
                b = tf.get_variable(
                    name='b', shape=n_filter, initializer=b_init, dtype=LayersConfig.tf_dtype, trainable=True,
                    **self.b_init_args
                )

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init:
            self._add_params([We, b])
        else:
            self._add_params(We)
