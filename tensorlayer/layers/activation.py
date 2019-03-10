#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.initializers import truncated_normal
# from tensorlayer.layers.core import LayersConfig

from tensorlayer.activation import leaky_relu6
from tensorlayer.activation import leaky_twice_relu6

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'PRelu',
    'PRelu6',
    'PTRelu6',
]


class PRelu(Layer):
    """
    The :class:`PRelu` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    channel_shared : boolean
        If True, single weight is shared by all channels.
    a_init : initializer
        The initializer for initializing the alpha(s).
    a_init_args : dictionary
        The arguments for initializing the alpha(s).
    name : None or str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
            self,
            channel_shared=False,
            a_init=truncated_normal(mean=0.0, stddev=0.1),
            a_init_args=None,
            name=None  # "prelu"
    ):

        # super(PRelu, self).__init__(prev_layer=prev_layer, act=tf.nn.leaky_relu, a_init_args=a_init_args, name=name)
        super(PRelu, self).__init__(name)
        self.channel_shared = channel_shared
        self.a_init = a_init
        self.a_init_args = a_init_args

        logging.info("PRelu %s: channel_shared: %s" % (self.name, self.channel_shared))

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = inputs_shape[-1]
        self.alpha_var = self._get_weights("alpha", shape=w_shape, init=self.a_init)
        # self.alpha_var = tf.compat.v1.get_variable(
        #     name=self.name + '/alpha', shape=w_shape, initializer=self.a_init, dtype=LayersConfig.tf_dtype,
        #     **self.a_init_args
        # )
        self.alpha_var_constrained = tf.nn.sigmoid(self.alpha_var, name="constraining_alpha_var_in_0_1")
        # self.add_weights(self.alpha_var)

    def forward(self, inputs):

        pos = tf.nn.relu(inputs)
        neg = -self.alpha_var_constrained * tf.nn.relu(-inputs)

        return pos + neg


class PRelu6(Layer):
    """
    The :class:`PRelu6` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This Layer is a modified version of the :class:`PRelu`.

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
    channel_shared : boolean
        If True, single weight is shared by all channels.
    a_init : initializer
        The initializer for initializing the alpha(s).
    a_init_args : dictionary
        The arguments for initializing the alpha(s).
    name : None or str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
            self,
            channel_shared=False,
            a_init=tf.compat.v1.initializers.truncated_normal(mean=0.0, stddev=0.1),
            a_init_args=None,
            name=None  # "prelu6"
    ):

        # super(PRelu6, self).__init__(prev_layer=prev_layer, act=leaky_relu6, a_init_args=a_init_args, name=name)
        super(PRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.a_init = a_init
        self.a_init_args = a_init_args

        logging.info("PRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = inputs_shape[-1]
        self.alpha_var = self._get_weights("alpha", shape=w_shape, init=self.a_init)
        # self.alpha_var = tf.compat.v1.get_variable(
        #     name=self.name + '/alpha', shape=w_shape, initializer=self.a_init, dtype=LayersConfig.tf_dtype,
        #     **self.a_init_args
        # )

        self.alpha_var_constrained = tf.nn.sigmoid(self.alpha_var, name="constraining_alpha_var_in_0_1")
        # self.add_weights(self.alpha_var)

    def forward(self, inputs):
        outputs = self._apply_activation(inputs, **{'alpha': self.alpha_var_constrained, 'name': "prelu6_activation"})
        return outputs


class PTRelu6(Layer):
    """
    The :class:`PTRelu6` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This Layer is a modified version of the :class:`PRelu`.

    This activation layer use a modified version :func:`tl.act.leaky_relu` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also use a modified version of the activation function :func:`tf.nn.relu6` introduced by the following paper:
    `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    This activation layer push further the logic by adding `leaky` behaviour both below zero and above six.

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x in [0, 6]: ``f(x) = x``.
      - When x > 6: ``f(x) = 6 + (alpha_high * (x-6))``.

    This version goes one step beyond :class:`PRelu6` by introducing leaky behaviour on the positive side when x > 6.

    Parameters
    ----------
    channel_shared : boolean
        If True, single weight is shared by all channels.
    a_init : initializer
        The initializer for initializing the alpha(s).
    a_init_args : dictionary
        The arguments for initializing the alpha(s).
    name : None or str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    def __init__(
            self,
            channel_shared=False,
            a_init=tf.compat.v1.initializers.truncated_normal(mean=0.0, stddev=0.1),
            a_init_args=None,
            name=None  # "ptreLU6"
    ):

        # super(PTRelu6, self).__init__(prev_layer=prev_layer, act=leaky_twice_relu6, a_init_args=a_init_args, name=name)
        super().__init__(name)
        self.channel_shared = channel_shared
        self.a_init = a_init
        self.a_init_args = a_init_args

        logging.info("PTRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = inputs_shape[-1]

        # Alpha for outputs lower than zeros
        self.alpha_low = self._get_weights("alpha_low", shape=w_shape, init=self.a_init, init_args=self.a_init_args)
        # self.alpha_low = tf.compat.v1.get_variable(
        #     name=self.name + '/alpha_low', shape=w_shape, initializer=self.a_init, dtype=LayersConfig.tf_dtype,
        #     **self.a_init_args
        # )
        self.alpha_low_constrained = tf.nn.sigmoid(self.alpha_low, name="constraining_alpha_low_in_0_1")

        # Alpha for outputs higher than 6
        self.alpha_high = self._get_weights("alpha_high", shape=w_shape, init=self.a_init, init_args=self.a_init_args)
        # self.alpha_high = tf.compat.v1.get_variable(
        #     name=self.name + '/alpha_high', shape=w_shape, initializer=self.a_init, dtype=LayersConfig.tf_dtype,
        #     **self.a_init_args
        # )

        self.alpha_high_constrained = tf.nn.sigmoid(self.alpha_high, name="constraining_alpha_high_in_0_1")

        # self.add_weights([self.alpha_low, self.alpha_high])

    def forward(self, inputs):
        outputs = self._apply_activation(
            inputs, **{
                'alpha_low': self.alpha_low_constrained,
                'alpha_high': self.alpha_high_constrained,
                'name': "ptrelu6_activation"
            }
        )
        return outputs
