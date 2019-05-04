#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import flatten_reshape

__all__ = [
    'Flatten',
    'Reshape',
    'Transpose',
    'Shuffle',
]


class Flatten(Layer):
    """A layer that reshapes high-dimension input into a vector.

    Then we often apply Dense, RNN, Concat and etc on the top of a flatten layer.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> x = tl.layers.Input([8, 4, 3], name='input')
    >>> y = tl.layers.Flatten(name='flatten')(x)
    [8, 12]

    """

    def __init__(self, name=None):  #'flatten'):
        super(Flatten, self).__init__(name)

        self.build()
        self._built = True

        logging.info("Flatten %s:" % (self.name))

    def __repr__(self):
        s = '{classname}('
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = flatten_reshape(inputs, name=self.name)
        return outputs


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
    >>> x = tl.layers.Input([8, 4, 3], name='input')
    >>> y = tl.layers.Reshape(shape=[-1, 12], name='reshape')(x)
    (8, 12)

    """

    def __init__(self, shape, name=None):  #'reshape'):
        super(Reshape, self).__init__(name)
        self.shape = shape

        logging.info("Reshape %s" % (self.name))

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'shape={shape},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = tf.reshape(inputs, shape=self.shape, name=self.name)
        return outputs


class Transpose(Layer):
    """A layer that transposes the dimension of a tensor.

    See `tf.transpose() <https://www.tensorflow.org/api_docs/python/tf/transpose>`__ .

    Parameters
    ----------
    perm: list of int
        The permutation of the dimensions, similar with ``numpy.transpose``.
        If None, it is set to (n-1...0), where n is the rank of the input tensor.
    conjugate: bool
        By default False. If True, returns the complex conjugate of complex numbers (and transposed)
        For example [[1+1j, 2+2j]] --> [[1-1j], [2-2j]]
    name : str
        A unique layer name.

    Examples
    ----------
    >>> x = tl.layers.Input([8, 4, 3], name='input')
    >>> y = tl.layers.Transpose(perm=[0, 2, 1], conjugate=False, name='trans')(x)
    (8, 3, 4)

    """

    def __init__(self, perm=None, conjugate=False, name=None):  #'transpose'):
        super(Transpose, self).__init__(name)
        self.perm = perm
        self.conjugate = conjugate

        logging.info("Transpose  %s: perm: %s, conjugate: %s" % (self.name, self.perm, self.conjugate))

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'perm={perm},'
        s += 'conjugate={conjugate},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = tf.transpose(a=inputs, perm=self.perm, conjugate=self.conjugate, name=self.name)
        return outputs


class Shuffle(Layer):
    """A layer that shuffle a 2D image [batch, height, width, channel], see `here <https://arxiv.org/abs/1707.01083>`__.

    Parameters
    ----------
    group: int
        The number of groups.
    name : str
        A unique layer name.

    Examples
    --------
    >>> x = tl.layers.Input([1, 16, 16, 8], name='input')
    >>> y = tl.layers.Shuffle(group=2, name='shuffle')(x)
    (1, 16, 16, 8)

    """

    def __init__(self, group, name=None):  #'reshape'):
        super(Shuffle, self).__init__(name)
        self.group = group

        logging.info("Shuffle %s" % (self.name))

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'group={group},'
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        in_shape = inputs.get_shape().as_list()
        h, w, in_channel = in_shape[1:]
        if in_channel % self.group != 0:
            raise ValueError(
                "The in_channel must be a multiple of the number of groups. The in_channel got %d and the number of groups is %d."
                % (in_channel, self.group)
            )
        temp = tf.reshape(inputs, [-1, h, w, in_channel // self.group, self.group])
        temp = tf.transpose(temp, [0, 1, 2, 4, 3])
        outputs = tf.reshape(temp, [-1, h, w, in_channel], name=self.name)
        return outputs
