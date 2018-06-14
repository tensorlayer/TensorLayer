#! /usr/bin/python
# -*- coding: utf-8 -*-
"""A file containing various activation functions."""

import tensorflow as tf

from tensorlayer.decorators import deprecated

__all__ = [
    'leaky_relu',
    'leaky_relu6',
    'leaky_twice_relu6',
    'lrelu',
    'lrelu6',
    'ltrelu6',
    'ramp',
    'swish',
    'sign',
    'htanh',
    'hard_tanh',
    'pixel_wise_softmax',
]


def ramp(x, v_min=0, v_max=1, name=None):
    """Ramp activation function.

    Parameters
    ----------
    x : Tensor
        input.
    v_min : float
        cap input to v_min as a lower bound.
    v_max : float
        cap input to v_max as a upper bound.
    name : str
        The function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)


@deprecated(date="2018-09-30", instructions="This API is deprecated. Please use as `tf.nn.leaky_relu`")
def leaky_relu(x, alpha=0.2, name="leaky_relu"):
    """leaky_relu can be used through its shortcut: :func:`tl.act.lrelu`.

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
    >>> import tensorlayer as tl
    >>> net = tl.layers.DenseLayer(net, 100, act=lambda x : tl.act.lrelu(x, 0.2), name='dense')

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    """

    if not (0 < alpha <= 1):
        raise ValueError("`alpha` value must be in [0, 1]`")

    with tf.name_scope(name, "leaky_relu") as name_scope:
        x = tf.convert_to_tensor(x, name="features")
        return tf.maximum(x, alpha * x, name=name_scope)


def leaky_relu6(x, alpha=0.2, name="leaky_relu6"):
    """:func:`leaky_relu6` can be used through its shortcut: :func:`tl.act.lrelu6`.

    This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also follows the behaviour of the activation function :func:`tf.nn.relu6` introduced by the following paper:
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
    >>> import tensorlayer as tl
    >>> net = tl.layers.DenseLayer(net, 100, act=lambda x : tl.act.leaky_relu6(x, 0.2), name='dense')

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__
    """

    if not isinstance(alpha, tf.Tensor) and not (0 < alpha <= 1):
        raise ValueError("`alpha` value must be in [0, 1]`")

    with tf.name_scope(name, "leaky_relu6") as name_scope:
        x = tf.convert_to_tensor(x, name="features")
        return tf.minimum(tf.maximum(x, alpha * x), 6, name=name_scope)


def leaky_twice_relu6(x, alpha_low=0.2, alpha_high=0.2, name="leaky_relu6"):
    """:func:`leaky_twice_relu6` can be used through its shortcut: :func:`:func:`tl.act.ltrelu6`.

    This activation function is a modified version :func:`leaky_relu` introduced by the following paper:
    `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__

    This activation function also follows the behaviour of the activation function :func:`tf.nn.relu6` introduced by the following paper:
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
    >>> import tensorlayer as tl
    >>> net = tl.layers.DenseLayer(net, 100, act=lambda x : tl.act.leaky_twice_relu6(x, 0.2, 0.2), name='dense')

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models [A. L. Maas et al., 2013] <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`__
    - `Convolutional Deep Belief Networks on CIFAR-10 [A. Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`__

    """

    if not isinstance(alpha_high, tf.Tensor) and not (0 < alpha_high <= 1):
        raise ValueError("`alpha_high` value must be in [0, 1]`")

    if not isinstance(alpha_low, tf.Tensor) and not (0 < alpha_low <= 1):
        raise ValueError("`alpha_low` value must be in [0, 1]`")

    with tf.name_scope(name, "leaky_twice_relu6") as name_scope:
        x = tf.convert_to_tensor(x, name="features")

        x_is_above_0 = tf.minimum(x, 6 * (1 - alpha_high) + alpha_high * x)
        x_is_below_0 = tf.minimum(alpha_low * x, 0)

        return tf.maximum(x_is_above_0, x_is_below_0, name=name_scope)


def swish(x, name='swish'):
    """Swish function.

     See `Swish: a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941>`__.

    Parameters
    ----------
    x : Tensor
        input.
    name: str
        function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    with tf.name_scope(name):
        x = tf.nn.sigmoid(x) * x
    return x


@tf.RegisterGradient("QuantizeGrad")
def _sign_grad(unused_op, grad):
    return tf.clip_by_value(grad, -1, 1)


def sign(x):
    """Sign function.

    Clip and binarize tensor using the straight through estimator (STE) for the gradient, usually be used for
    quantizing values in `Binarized Neural Networks`: https://arxiv.org/abs/1602.02830.

    Parameters
    ----------
    x : Tensor
        input.

    Examples
    --------
    >>> net = tl.layers.DenseLayer(net, 100, act=lambda x : tl.act.lrelu(x, 0.2), name='dense')

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    References
    ----------
    - `Rectifier Nonlinearities Improve Neural Network Acoustic Models, Maas et al. (2013)`
       http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf

    - `BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, \
       Courbariaux et al. (2016)` https://arxiv.org/abs/1602.02830

    """
    with tf.get_default_graph().gradient_override_map({"sign": "QuantizeGrad"}):
        return tf.sign(x, name='sign')


# if tf.__version__ > "1.7":
#     @tf.custom_gradient
#     def sign(x):  # https://www.tensorflow.org/versions/master/api_docs/python/tf/custom_gradient?hl=ES#top_of_page
#         """Differentiable sign function using sigmoid as the derivation function,
#         see `tf.sign <https://www.tensorflow.org/api_docs/python/tf/sign>`__ and `tf.custom_gradient
#         <https://www.tensorflow.org/versions/master/api_docs/python/tf/custom_gradient?hl=ES#top_of_page>`__.
#
#         Parameters
#         ----------
#         x : Tensor
#             input.
#
#         Returns
#         -------
#         Tensor
#             A ``Tensor`` in the same type as ``x``.
#
#         """
#         tao = tf.nn.sigmoid(x)
#         def grad():
#             return tao * (1 - tao)
#         return tf.sign(x), grad


def hard_tanh(x, name='htanh'):
    """Hard tanh activation function.

    Which is a ramp function with low bound of -1 and upper bound of 1, shortcut is `htanh`.

    Parameters
    ----------
    x : Tensor
        input.
    name : str
        The function name (optional).

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    """
    # with tf.variable_scope("hard_tanh"):
    return tf.clip_by_value(x, -1, 1, name=name)


@deprecated(date="2018-06-30", instructions="This API will be deprecated soon as tf.nn.softmax can do the same thing")
def pixel_wise_softmax(x, name='pixel_wise_softmax'):
    """Return the softmax outputs of images, every pixels have multiple label, the sum of a pixel is 1.

    Usually be used for image segmentation.

    Parameters
    ----------
    x : Tensor
        input.
            - For 2d image, 4D tensor (batch_size, height, weight, channel), where channel >= 2.
            - For 3d image, 5D tensor (batch_size, depth, height, weight, channel), where channel >= 2.
    name : str
        function name (optional)

    Returns
    -------
    Tensor
        A ``Tensor`` in the same type as ``x``.

    Examples
    --------
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - dice_coe(outputs, y_, epsilon=1e-5)

    References
    ----------
    - `tf.reverse <https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#reverse>`__

    """
    with tf.name_scope(name):
        return tf.nn.softmax(x)


# Alias
lrelu = leaky_relu
lrelu6 = leaky_relu6
ltrelu6 = leaky_twice_relu6
htanh = hard_tanh
