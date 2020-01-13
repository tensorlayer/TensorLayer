#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

__all__ = [
    'ExpandDims',
    'Tile',
]


class ExpandDims(Layer):
    """
    The :class:`ExpandDims` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/tf/expand_dims>`__ .

    Parameters
    ----------
    axis : int
        The dimension index at which to expand the shape of input.
    name : str
        A unique layer name. If None, a unique name will be automatically assigned.

    Examples
    --------
    >>> x = tl.layers.Input([10, 3], name='in')
    >>> y = tl.layers.ExpandDims(axis=-1)(x)
    [10, 3, 1]
    """

    def __init__(
            self,
            axis,
            name=None  # 'expand_dims',
    ):
        super(ExpandDims, self).__init__(name)
        self.axis = axis

        self.build((None, ))
        self._built = True

        logging.info("ExpandDims  %s: axis: %d" % (self.name, self.axis))

    def __repr__(self):
        s = '{classname}('
        s += 'axis={axis},'
        s += 'name={name}'
        s += ")"
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = tf.expand_dims(inputs, axis=self.axis, name=self.name)
        return outputs


class Tile(Layer):
    """
    The :class:`Tile` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/tf/tile>`__ .

    Parameters
    ----------
    multiples: tensor
        Must be one of the following types: int32, int64.
        1-D Length must be the same as the number of dimensions in input.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> x = tl.layers.Input([10, 3], name='in')
    >>> y = tl.layers.Tile(multiples=[2, 3])(x)
    [20, 9]
    """

    def __init__(self, multiples=None, name=None):  #'tile'):

        super(Tile, self).__init__(name)
        self.multiples = multiples

        self.build((None, ))
        self._built = True

        logging.info("Tile  %s: multiples: %s" % (self.name, self.multiples))

    def __repr__(self):
        s = '{classname}('
        s += 'multiples={multiples},'
        s += 'name={name}'
        s += ")"
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = tf.tile(inputs, multiples=self.multiples, name=self.name)
        return outputs
