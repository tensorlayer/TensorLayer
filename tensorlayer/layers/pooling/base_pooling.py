#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

__all__ = ['PoolLayer']


class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer.
    You can choose ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D input or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D input.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    def __init__(
        self,
        prev_layer,
        ksize=(1, 2, 2, 1),
        strides=(1, 2, 2, 1),
        padding='SAME',
        pool=tf.nn.max_pool,
        name='pool_layer',
    ):
        super(PoolLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "PoolLayer %s: ksize: %s strides: %s padding: %s pool: %s" %
            (self.name, str(ksize), str(strides), padding, pool.__name__)
        )

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)

        self._add_layers(self.outputs)
