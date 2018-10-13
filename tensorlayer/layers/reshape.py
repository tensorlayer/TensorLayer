#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.layers.utils.reshape import flatten_reshape

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'Flatten',
    'Reshape',
    'Transpose',
]


class Flatten(Layer):
    """A layer that reshapes high-dimension input into a vector.

    Then we often apply Dense, RNN, Concat and etc on the top of a flatten layer.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> net = tl.layers.Input(name='input')(x)
    >>> net = tl.layers.Flatten(name='flatten')(net)
    [?, 784]
    """

    def __init__(self, name='flatten'):

        self.name = name

        super(Flatten, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = flatten_reshape(self._temp_data['inputs'], name=self.name)


class Reshape(Layer):
    """A layer that reshapes a given tensor.

    Parameters
    ----------
    shape : tuple of int
        The output shape, see ``tf.reshape``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(None, 784))
    >>> net = tl.layers.Input(name='input')(x)
    >>> net = tl.layers.Reshape([-1, 28, 28, 1], name='reshape')(net)
    >>> print(net.outputs)
    (?, 28, 28, 1)

    """

    def __init__(self, shape, name='reshape'):

        if not shape:
            raise ValueError("Shape list can not be empty")

        self.shape = shape
        self.name = name

        super(Reshape, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = tf.reshape(self._temp_data['inputs'], shape=self.shape, name=self.name)


class Transpose(Layer):
    """A layer that transposes the dimension of a tensor.

    See `tf.transpose() <https://www.tensorflow.org/api_docs/python/tf/transpose>`__ .

    Parameters
    ----------
    perm: list of int
        The permutation of the dimensions, similar with ``numpy.transpose``.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> net = tl.layers.Input(name='input')(x)
    >>> net = tl.layers.Transpose(perm=[0, 1, 3, 2], name='trans')(net)
    [None, 28, 1, 28]

    """

    def __init__(self, perm=None, name='transpose'):

        if perm is None:
            raise AssertionError("The `perm` argument cannot be None")

        self.perm = perm
        self.name = name

        super(Transpose, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("perm: %s" % self.perm)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = tf.transpose(self._temp_data['inputs'], perm=self.perm, name=self.name)
