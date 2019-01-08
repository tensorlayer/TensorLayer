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
    name : None or str
        A unique layer name.
    """

    def __init__(
            self,
            n_filter=32,
            filter_size=(3, 3),
            strides=(2, 2),
            n_group=2,
            act=None,
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None,  #'groupconv',
    ):  # Windaway

        # super(GroupConv2d, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.n_group = n_group
        self.act = act
        self.padding = padding
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args
        logging.info(
            "GroupConv2d %s: n_filter: %d size: %s strides: %s n_group: %d pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), n_group, padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs):
        self.groupConv = lambda i, k: tf.nn.conv2d(
            i, k, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding
        )
        channels = int(inputs.get_shape()[-1])

        self.We = tf.get_variable(
            name=self.name + '\W',
            shape=[self.filter_size[0], self.filter_size[1], channels / self.n_group, self.n_filter],
            initializer=self.W_init, dtype=LayersConfig.tf_dtype, trainable=True, **self.W_init_args
        )
        if self.b_init:
            self.b = tf.get_variable(
                name=self.name + '\b', shape=self.n_filter, initializer=self.b_init, dtype=LayersConfig.tf_dtype,
                trainable=True, **self.b_init_args
            )
            self.add_weights([self.We, self.b])
        else:
            self.add_weights(self.We)

    def forward(self, inputs):
        if self.n_group == 1:
            outputs = self.groupConv(inputs, self.We)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self.inputs)
            weightsGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self.We)
            convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]
            outputs = tf.concat(axis=3, values=convGroups)
        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
        if self.act:
            outputs = self.act(outputs)
        return outputs
