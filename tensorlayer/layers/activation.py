#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.activation import leaky_relu6
from tensorlayer.activation import leaky_twice_relu6

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'PReluLayer',
    'PRelu6Layer',
    'PTRelu6Layer',
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
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, channel_shared=False, a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            a_init_args=None, name="PReluLayer"
    ):

        super(PReluLayer,
              self).__init__(prev_layer=prev_layer, act=tf.nn.leaky_relu, a_init_args=a_init_args, name=name)

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

            alpha_var_constrained = tf.nn.sigmoid(alpha_var, name="constraining_alpha_var_in_0_1")

        self.outputs = self._apply_activation(
            self.inputs, **{
                'alpha': alpha_var_constrained,
                'name': "PReLU_activation"
            }
        )

        self._add_layers(self.outputs)
        self._add_params(alpha_var)


class PRelu6Layer(Layer):
    """
    The :class:`PRelu6Layer` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This Layer is a modified version of the :class:`PReluLayer`.

    This activation layer use a modified version :func:`tl.act.leaky_relu` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also use a modified version of the activation function :func:`tf.nn.relu6` introduced by the following paper:
    `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    This activation layer push further the logic by adding `leaky` behaviour both below zero and above six.

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x in [0, 6]: ``f(x) = x``.
      - When x > 6: ``f(x) = 6``.

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
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, channel_shared=False, a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            a_init_args=None, name="PReLU6_layer"
    ):

        super(PRelu6Layer, self).__init__(prev_layer=prev_layer, act=leaky_relu6, a_init_args=a_init_args, name=name)

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

            alpha_var_constrained = tf.nn.sigmoid(alpha_var, name="constraining_alpha_var_in_0_1")

        self.outputs = self._apply_activation(
            self.inputs, **{
                'alpha': alpha_var_constrained,
                'name': "PReLU6_activation"
            }
        )

        self._add_layers(self.outputs)
        self._add_params(alpha_var)


class PTRelu6Layer(Layer):
    """
    The :class:`PTRelu6Layer` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This Layer is a modified version of the :class:`PReluLayer`.

    This activation layer use a modified version :func:`tl.act.leaky_relu` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also use a modified version of the activation function :func:`tf.nn.relu6` introduced by the following paper:
    `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    This activation layer push further the logic by adding `leaky` behaviour both below zero and above six.

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x in [0, 6]: ``f(x) = x``.
      - When x > 6: ``f(x) = 6 + (alpha_high * (x-6))``.

    This version goes one step beyond :class:`PRelu6Layer` by introducing leaky behaviour on the positive side when x > 6.

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
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, channel_shared=False, a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            a_init_args=None, name="PTReLU6_layer"
    ):

        super(PTRelu6Layer,
              self).__init__(prev_layer=prev_layer, act=leaky_twice_relu6, a_init_args=a_init_args, name=name)

        if channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        logging.info("PTRelu6Layer %s: channel_shared: %s" % (self.name, channel_shared))

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name):

            # Alpha for outputs lower than zeros
            alpha_low = tf.get_variable(
                name='alpha_low', shape=w_shape, initializer=a_init, dtype=LayersConfig.tf_dtype, **self.a_init_args
            )

            alpha_low_constrained = tf.nn.sigmoid(alpha_low, name="constraining_alpha_low_in_0_1")

            # Alpha for outputs higher than 6
            alpha_high = tf.get_variable(
                name='alpha_high', shape=w_shape, initializer=a_init, dtype=LayersConfig.tf_dtype, **self.a_init_args
            )

            alpha_high_constrained = tf.nn.sigmoid(alpha_high, name="constraining_alpha_high_in_0_1")

        self.outputs = self.outputs = self._apply_activation(
            self.inputs, **{
                'alpha_low': alpha_low_constrained,
                'alpha_high': alpha_high_constrained,
                'name': "PTReLU6_activation"
            }
        )

        self._add_layers(self.outputs)
        self._add_params([alpha_low, alpha_high])
