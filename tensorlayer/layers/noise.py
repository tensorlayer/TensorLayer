#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'GaussianNoiseLayer',
]


class GaussianNoiseLayer(Layer):
    """
    The :class:`GaussianNoiseLayer` class is noise layer that adding noise with
    gaussian distribution to the activation.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    mean : float
        The mean. Default is 0.
    stddev : float
        The standard deviation. Default is 1.
    is_train : boolean
        Is trainable layer. If False, skip this layer. default is True.
    seed : int or None
        The seed for random noise.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(100, 784))
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, name='dense3')
    >>> net = tl.layers.GaussianNoiseLayer(net, name='gaussian')
    (64, 100)

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        mean=0.0,
        stddev=1.0,
        is_train=True,
        seed=None,
        name='gaussian_noise_layer',
    ):

        self.prev_layer = prev_layer
        self.mean = mean
        self.stddev = stddev
        self.is_train = is_train
        self.seed = seed
        self.name = name

        super(GaussianNoiseLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("mean: %s" % self.mean)
        except AttributeError:
            pass

        try:
            additional_str.append("stddev: %s" % self.stddev)
        except AttributeError:
            pass

        return self._str(additional_str)

    def compile(self, prev_layer, is_train=True):

        super(GaussianNoiseLayer, self).compile(prev_layer)

        if is_train is False:
            logging.info("  -> [Not Training] - skip `%s`" % self.__class__.__name__)
            self.outputs = self.inputs

        else:
            with tf.variable_scope(self.name):
                noise = tf.random_normal(
                    shape=self.inputs.get_shape(),
                    mean=self.mean,
                    stddev=self.stddev,
                    seed=self.seed,
                    dtype=self.inputs.dtype
                )
                self.outputs = tf.add(self.inputs, noise)

        self._add_layers(self.outputs)
