#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer import logging
import tensorlayer as tl
from tensorlayer.initializers import truncated_normal
from tensorlayer.layers.core import Module

__all__ = [
    'PRelu',
    'PRelu6',
    'PTRelu6',
]


class PRelu(Module):
    """
    The :class:`PRelu` class is Parametric Rectified Linear layer.
    It follows f(x) = alpha * x for x < 0, f(x) = x for x >= 0,
    where alpha is a learned array with the same shape as x.

    Parameters
    ----------
    channel_shared : boolean
        If True, single weight is shared by all channels.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    a_init : initializer
        The initializer for initializing the alpha(s).
    name : None or str
        A unique layer name.

    Examples
    -----------
    >>> inputs = tl.layers.Input([10, 5])
    >>> prelulayer = tl.layers.PRelu(channel_shared=True)
    >>> print(prelulayer)
    PRelu(channel_shared=True,in_channels=None,name=prelu)
    >>> prelu = prelulayer(inputs)
    >>> model = tl.models.Model(inputs=inputs, outputs=prelu)
    >>> out = model(data, is_train=True)

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/abs/1502.01852>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    def __init__(
        self, channel_shared=False, in_channels=None, a_init=truncated_normal(mean=0.0, stddev=0.05), name=None,
        data_format='channels_last', dim=2
    ):

        super(PRelu, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.a_init = a_init
        self.data_format = data_format
        self.dim = dim

        if self.channel_shared:
            self.build((None, ))
            self._built = True
        elif self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info("PRelu %s: channel_shared: %s" % (self.name, self.channel_shared))

    def __repr__(self):
        s = ('{classname}(')
        s += 'channel_shared={channel_shared},'
        s += 'in_channels={in_channels},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        elif self.data_format == 'channels_last':
            w_shape = (self.in_channels, )
        elif self.data_format == 'channels_first':
            if self.dim == 2:
                w_shape = (1, self.in_channels, 1, 1)
            elif self.dim == 1:
                w_shape = (1, self.in_channels, 1)
            elif self.dim == 3:
                w_shape = (1, self.in_channels, 1, 1, 1)
            else:
                raise Exception("Dim should be equal to 1, 2 or 3")
        self.alpha_var = self._get_weights("alpha", shape=w_shape, init=self.a_init)
        self.relu = tl.ops.ReLU()
        self.sigmoid = tl.ops.Sigmoid()

    def forward(self, inputs):
        pos = self.relu(inputs)
        alpha_var_constrained = self.sigmoid(self.alpha_var)
        neg = -alpha_var_constrained * self.relu(-inputs)
        return pos + neg


class PRelu6(Module):
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
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    a_init : initializer
        The initializer for initializing the alpha(s).
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
        in_channels=None,
        a_init=truncated_normal(mean=0.0, stddev=0.05),
        name=None,  # "prelu6"
        data_format='channels_last',
        dim=2
    ):

        super(PRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.a_init = a_init
        self.data_format = data_format
        self.dim = dim

        if self.channel_shared:
            self.build((None, ))
            self._built = True
        elif self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info("PRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def __repr__(self):
        s = ('{classname}(')
        s += 'channel_shared={channel_shared},'
        s += 'in_channels={in_channels},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        elif self.data_format == 'channels_last':
            w_shape = (self.in_channels, )
        elif self.data_format == 'channels_first':
            if self.dim == 2:
                w_shape = (1, self.in_channels, 1, 1)
            elif self.dim == 1:
                w_shape = (1, self.in_channels, 1)
            elif self.dim == 3:
                w_shape = (1, self.in_channels, 1, 1, 1)
            else:
                raise Exception("Dim should be equal to 1, 2 or 3")
        self.alpha_var = self._get_weights("alpha", shape=w_shape, init=self.a_init)
        self.sigmoid = tl.ops.Sigmoid()
        self.relu = tl.ops.ReLU()

    # @tf.function
    def forward(self, inputs):
        alpha_var_constrained = self.sigmoid(self.alpha_var)
        pos = self.relu(inputs)
        pos_6 = -self.relu(inputs - 6)
        neg = -alpha_var_constrained * self.relu(-inputs)
        return pos + pos_6 + neg


class PTRelu6(Module):
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
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    a_init : initializer
        The initializer for initializing the alpha(s).
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
        in_channels=None,
        a_init=truncated_normal(mean=0.0, stddev=0.05),
        name=None  # "ptrelu6"
    ):

        super(PTRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.a_init = a_init

        if self.channel_shared:
            self.build((None, ))
            self._built = True
        elif self.in_channels:
            self.build((None, self.in_channels))
            self._built = True

        logging.info("PTRelu6 %s: channel_shared: %s" % (self.name, self.channel_shared))

    def __repr__(self):
        s = ('{classname}(')
        s += 'channel_shared={channel_shared},'
        s += 'in_channels={in_channels},'
        s += 'name={name}'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.channel_shared:
            w_shape = (1, )
        elif self.data_format == 'channels_last':
            w_shape = (self.in_channels, )
        elif self.data_format == 'channels_first':
            if self.dim == 2:
                w_shape = (1, self.in_channels, 1, 1)
            elif self.dim == 1:
                w_shape = (1, self.in_channels, 1)
            elif self.dim == 3:
                w_shape = (1, self.in_channels, 1, 1, 1)
            else:
                raise Exception("Dim should be equal to 1, 2 or 3")

        # Alpha for outputs lower than zeros
        self.alpha_low = self._get_weights("alpha_low", shape=w_shape, init=self.a_init)
        self.sigmoid = tl.ops.Sigmoid()
        self.relu = tl.ops.ReLU()
        # Alpha for outputs higher than 6
        self.alpha_high = self._get_weights("alpha_high", shape=w_shape, init=self.a_init)

    # @tf.function
    def forward(self, inputs):
        alpha_low_constrained = self.sigmoid(self.alpha_low)
        alpha_high_constrained = self.sigmoid(self.alpha_high)
        pos = self.relu(inputs)
        pos_6 = -self.relu(inputs - 6) + alpha_high_constrained * self.relu(inputs - 6)
        neg = -alpha_low_constrained * self.relu(-inputs)

        return pos + pos_6 + neg
