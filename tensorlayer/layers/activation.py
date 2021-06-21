#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer import logging
import tensorlayer as tl
from tensorlayer.initializers import truncated_normal
from tensorlayer.layers.core import Module

__all__ = [
    'PRelu', 'PRelu6', 'PTRelu6', 'LeakyReLU', 'LeakyReLU6', 'LeakyTwiceRelu6', 'Ramp', 'Swish', 'HardTanh', 'Mish'
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
    >>> prelulayer = tl.layers.PRelu(channel_shared=True, in_channels=5)(inputs)

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
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        pos = self.relu(inputs)
        self.alpha_var_constrained = self.sigmoid(self.alpha_var)
        neg = -self.alpha_var_constrained * self.relu(-inputs)
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

    Examples
    -----------
    >>> inputs = tl.layers.Input([10, 5])
    >>> prelulayer = tl.layers.PRelu6(channel_shared=True, in_channels=5)(inputs)

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
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

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

    Examples
    -----------
    >>> inputs = tl.layers.Input([10, 5])
    >>> prelulayer = tl.layers.PTRelu6(channel_shared=True, in_channels=5)(inputs)

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
        data_format='channels_last',
        a_init=truncated_normal(mean=0.0, stddev=0.05),
        name=None  # "ptrelu6"
    ):

        super(PTRelu6, self).__init__(name)
        self.channel_shared = channel_shared
        self.in_channels = in_channels
        self.data_format = data_format
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
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        alpha_low_constrained = self.sigmoid(self.alpha_low)
        alpha_high_constrained = self.sigmoid(self.alpha_high)
        pos = self.relu(inputs)
        pos_6 = -self.relu(inputs - 6) + alpha_high_constrained * self.relu(inputs - 6)
        neg = -alpha_low_constrained * self.relu(-inputs)

        return pos + pos_6 + neg


class Ramp(Module):
    """Ramp activation function.

        Reference: [tf.clip_by_value]<https://www.tensorflow.org/api_docs/python/tf/clip_by_value>

        Parameters
        ----------
        x : Tensor
            input.
        v_min : float
            cap input to v_min as a lower bound.
        v_max : float
            cap input to v_max as a upper bound.

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        Examples
        -----------
        >>> inputs = tl.layers.Input([10, 5])
        >>> prelulayer = tl.layers.Ramp()(inputs)

        """

    def __init__(self, v_min=0, v_max=1):
        super(Ramp, self).__init__()
        self._built = True
        self.v_min = v_min
        self.v_max = v_max

    def forward(self, x):
        return tl.ops.clip_by_value(x, clip_value_min=self.v_min, clip_value_max=self.v_max)


class LeakyReLU(Module):
    """

    This function is a modified version of ReLU, introducing a nonzero gradient for negative input. Introduced by the paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    The function return the following results:
      - When x < 0: ``f(x) = alpha_low * x``.
      - When x >= 0: ``f(x) = x``.

    Parameters
    ----------
    x : Tensor
        Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.
    alpha : float
        Slope.
    name : str
        The function name (optional).

    Examples
    --------
    >>> net = tl.layers.Input([10, 200])
    >>> net = tl.layers.LeakyReLU(alpha=0.5)(net)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__()
        self._built = True
        self.alpha = alpha
        self._leakyrelu = tl.ops.LeakyReLU(alpha=alpha)

    def forward(self, x):
        return self._leakyrelu(x)


class LeakyReLU6(Module):
    """
        This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
        `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

        This activation function also follows the behaviour of the activation function :func:`tf.ops.relu6` introduced by the following paper:
        `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

        The function return the following results:
          - When x < 0: ``f(x) = alpha_low * x``.
          - When x in [0, 6]: ``f(x) = x``.
          - When x > 6: ``f(x) = 6``.

        Parameters
        ----------
        x : Tensor
            Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.
        alpha : float
            Slope.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tl.layers.Input([10, 200])
        >>> net = tl.layers.LeakyReLU6(alpha=0.5)(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        References
        ----------
        - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
        - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
        """

    def __init__(self, alpha=0.2):
        super(LeakyReLU6, self).__init__()
        self._built = True
        if not (0 < alpha <= 1):
            raise ValueError("`alpha` value must be in [0, 1]`")

        self.alpha = alpha
        self.minimum = tl.ops.Minimum()
        self.maximum = tl.ops.Maximum()

    def forward(self, x):
        return self.minimum(self.maximum(x, self.alpha * x), 6)


class LeakyTwiceRelu6(Module):
    """

        This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
        `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

        This activation function also follows the behaviour of the activation function :func:`tf.ops.relu6` introduced by the following paper:
        `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

        This function push further the logic by adding `leaky` behaviour both below zero and above six.

        The function return the following results:
          - When x < 0: ``f(x) = alpha_low * x``.
          - When x in [0, 6]: ``f(x) = x``.
          - When x > 6: ``f(x) = 6 + (alpha_high * (x-6))``.

        Parameters
        ----------
        x : Tensor
            Support input type ``float``, ``double``, ``int32``, ``int64``, ``uint8``, ``int16``, or ``int8``.
        alpha_low : float
            Slope for x < 0: ``f(x) = alpha_low * x``.
        alpha_high : float
            Slope for x < 6: ``f(x) = 6 (alpha_high * (x-6))``.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tl.layers.Input([10, 200])
        >>> net = tl.layers.LeakyTwiceRelu6(alpha_low=0.5, alpha_high=0.2)(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        References
        ----------
        - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
        - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

        """

    def __init__(self, alpha_low=0.2, alpha_high=0.2):
        super(LeakyTwiceRelu6, self).__init__()
        self._built = True
        if not (0 < alpha_high <= 1):
            raise ValueError("`alpha_high` value must be in [0, 1]`")

        if not (0 < alpha_low <= 1):
            raise ValueError("`alpha_low` value must be in [0, 1]`")

        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.minimum = tl.ops.Minimum()
        self.maximum = tl.ops.Maximum()

    def forward(self, x):
        x_is_above_0 = self.minimum(x, 6 * (1 - self.alpha_high) + self.alpha_high * x)
        x_is_below_0 = self.minimum(self.alpha_low * x, 0)
        return self.maximum(x_is_above_0, x_is_below_0)


class Swish(Module):
    """Swish function.

         See `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`__.

        Parameters
        ----------
        x : Tensor
            input.
        name: str
            function name (optional).

        Examples
        --------
        >>> net = tl.layers.Input([10, 200])
        >>> net = tl.layers.Swish()(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        """

    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = tl.ops.Sigmoid()
        self._built = True

    def forward(self, x):
        return self.sigmoid(x) * x


class HardTanh(Module):
    """Hard tanh activation function.

        Which is a ramp function with low bound of -1 and upper bound of 1, shortcut is `htanh`.

        Parameters
        ----------
        x : Tensor
            input.
        name : str
            The function name (optional).

        Examples
        --------
        >>> net = tl.layers.Input([10, 200])
        >>> net = tl.layers.HardTanh()(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        """

    def __init__(self):
        super(HardTanh, self).__init__()
        self._built = True

    def forward(self, x):
        return tl.ops.clip_by_value(x, -1, 1)


class Mish(Module):
    """Mish activation function.

        Reference: [Mish: A Self Regularized Non-Monotonic Neural Activation Function .Diganta Misra, 2019]<https://arxiv.org/abs/1908.08681>

        Parameters
        ----------
        x : Tensor
            input.

        Examples
        --------
        >>> net = tl.layers.Input([10, 200])
        >>> net = tl.layers.Mish()(net)

        Returns
        -------
        Tensor
            A ``Tensor`` in the same type as ``x``.

        """

    def __init__(self):
        super(Mish, self).__init__()
        self._tanh = tl.ops.Tanh()
        self._softplus = tl.ops.Softplus()
        self._built = True

    def forward(self, x):
        return x * self._tanh(self._softplus(x))
