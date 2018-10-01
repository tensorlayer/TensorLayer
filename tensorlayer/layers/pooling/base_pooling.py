#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['PoolLayer']


class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer.
    You can choose ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D input or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D input.

    Parameters
    ----------
    ksize : tuple of int
        The size of the window for each dimension of the input tensor.
        Note that: len(ksize) >= 4.
    strides : tuple of int
        The stride of the sliding window for each dimension of the input tensor.
        Note that: len(strides) >= 4.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    pool : pooling function
        One of ``tf.nn.max_pool``, ``tf.nn.avg_pool``, ``tf.nn.max_pool3d`` and ``f.nn.avg_pool3d``.
        See `TensorFlow pooling APIs <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`__
    name : str
        A unique layer name.

    Examples
    --------
    - see :class:`Conv2dLayer`.

    """

    def __init__(
        self,
        ksize=(1, 2, 2, 1),
        strides=(1, 2, 2, 1),
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool_layer',
    ):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.pool = pool
        self.name = name

        super(PoolLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("ksize: %s" % str(self.ksize))
        except AttributeError:
            pass

        try:
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("pool func: %s" % self.pool.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):

            self._temp_data['outputs'] = self.pool(
                self._temp_data['inputs'],
                ksize=self.ksize,
                strides=self.strides,
                padding=self.padding,
                name="pooling_op"
            )
