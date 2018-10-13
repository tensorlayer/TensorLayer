#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.activation import leaky_relu6
from tensorlayer.activation import leaky_twice_relu6

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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
    name : str
        A unique layer name.

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
        self,
        channel_shared=False,
        a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
        a_init_args=None,
        name="prelu"
    ):

        self.channel_shared = channel_shared
        self.a_init = a_init
        self.act = tf.nn.leaky_relu
        self.name = name

        super(PRelu, self).__init__(a_init_args=a_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("channel_shared: %s" % self.channel_shared)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self._temp_data['inputs'].get_shape()[-1])

        with tf.variable_scope(self.name):
            alpha_var = self._get_tf_variable(
                name='alpha',
                shape=w_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.a_init,
                **self.a_init_args
            )

            alpha_var_constrained = tf.nn.sigmoid(alpha_var, name="constraining_alpha_var_in_0_1")

        self._temp_data['outputs'] = self._apply_activation(
            self._temp_data['inputs'], **{
                'alpha': alpha_var_constrained,
                'name': "PReLU_activation"
            }
        )


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
    name : str
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
        a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
        a_init_args=None,
        name="prelu6"
    ):

        self.channel_shared = channel_shared
        self.a_init = a_init
        self.act = leaky_relu6
        self.name = name

        super(PRelu6, self).__init__(a_init_args=a_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("channel_shared: %s" % self.channel_shared)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self._temp_data['inputs'].get_shape()[-1])

        with tf.variable_scope(self.name):
            alpha_var = self._get_tf_variable(
                name='alpha',
                shape=w_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.a_init,
                **self.a_init_args
            )

            alpha_var_constrained = tf.nn.sigmoid(alpha_var, name="constraining_alpha_var_in_0_1")

        self._temp_data['outputs'] = self._apply_activation(
            self._temp_data['inputs'], **{
                'alpha': alpha_var_constrained,
                'name': "PReLU6_activation"
            }
        )


class PTRelu6(Layer):
    """
    The :class:`PTRelu6` class is Parametric Rectified Linear layer integrating ReLU6 behaviour.

    This layer is a modified version of the :class:`PRelu`.

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
    name : str
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
        a_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
        a_init_args=None,
        name="ptrelu6"
    ):

        self.channel_shared = channel_shared
        self.a_init = a_init
        self.act = leaky_twice_relu6
        self.name = name

        super(PTRelu6, self).__init__(a_init_args=a_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("channel_shared: %s" % self.channel_shared)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self.channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self._temp_data['inputs'].get_shape()[-1])

        with tf.variable_scope(self.name):

            # Alpha for outputs lower than zeros
            alpha_low = self._get_tf_variable(
                name='alpha_low',
                shape=w_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.a_init,
                **self.a_init_args
            )

            alpha_low_constrained = tf.nn.sigmoid(alpha_low, name="constraining_alpha_low_in_0_1")

            # Alpha for outputs higher than 6
            alpha_high = self._get_tf_variable(
                name='alpha_high',
                shape=w_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.a_init,
                **self.a_init_args
            )

            alpha_high_constrained = tf.nn.sigmoid(alpha_high, name="constraining_alpha_high_in_0_1")

        self._temp_data['outputs'] = self._apply_activation(
            self._temp_data['inputs'], **{
                'alpha_low': alpha_low_constrained,
                'alpha_high': alpha_high_constrained,
                'name': "PTReLU6_activation"
            }
        )
