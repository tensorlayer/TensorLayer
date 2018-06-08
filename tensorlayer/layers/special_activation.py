#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.activation import leaky_relu6

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'PReluLayer',
    'PRelu6Layer',
]


class PReluLayer(Layer):
    """
    The :class:`PReluLayer` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    channel_shared : boolean
        If True, single weight is shared by all channels.
    a_init : initializer
        The initializer for initializing the alpha(s).
    a_init_args : dictionary
        The arguments for initializing the alpha(s).
    name : str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, channel_shared=False, a_init=tf.constant_initializer(value=0.2), a_init_args=None,
            name="PReluLayer"
    ):

        super(PReluLayer, self).__init__(prev_layer=prev_layer, a_init_args=a_init_args, name=name)

        if channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        logging.info("PReluLayer %s: channel_shared: %s" % (self.name, channel_shared))

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name):
            alpha_var = tf.get_variable(
                name='alpha', shape=w_shape, initializer=a_init, dtype=LayersConfig.tf_dtype, **self.a_init_args
            )

        self.outputs = tf.nn.leaky_relu(self.inputs, alpha=alpha_var, name="PReLU_activation")

        self._add_layers(self.outputs)
        self._add_params(alpha_var)


class PRelu6Layer(Layer):
    """
    The :class:`PRelu6Layer` class is Parametric Rectified Linear layer.

    Modified version of PReLU Layer following the general idea of `ReLU6` introduced by the following paper:

    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    channel_shared : boolean
        If True, single weight is shared by all channels.
    a_init : initializer
        The initializer for initializing the alpha(s).
    a_init_args : dictionary
        The arguments for initializing the alpha(s).
    name : str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, channel_shared=False, a_init=tf.constant_initializer(value=0.2), a_init_args=None,
            name="PReLU6_layer"
    ):

        super(PRelu6Layer, self).__init__(prev_layer=prev_layer, a_init_args=a_init_args, name=name)

        if channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        logging.info("PRelu6Layer %s: channel_shared: %s" % (self.name, channel_shared))

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name):
            alpha_var = tf.get_variable(
                name='alpha', shape=w_shape, initializer=a_init, dtype=LayersConfig.tf_dtype, **self.a_init_args
            )

        self.outputs = leaky_relu6(self.inputs, alpha=alpha_var, name="PReLU6_activation")

        self._add_layers(self.outputs)
        self._add_params(alpha_var)
