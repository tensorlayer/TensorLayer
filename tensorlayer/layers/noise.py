#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'GaussianNoise',
]


class GaussianNoise(Layer):
    """
    The :class:`GaussianNoise` class is noise layer that adding noise with
    gaussian distribution to the activation.

    Parameters
    ------------
    mean : float
        The mean. Default is 0.
    stddev : float
        The standard deviation. Default is 1.
    seed : int or None
        The seed for random noise.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(100, 784))
    >>> net = tl.layers.Input(name='input')(x)
    >>> net = tl.layers.Dense(n_units=100, act=tf.nn.relu, name='dense3')(net)
    >>> net = tl.layers.GaussianNoise(name='gaussian')(net)
    (64, 100)

    """

    def __init__(
        self,
        mean=0.0,
        stddev=1.0,
        seed=None,
        name='gaussian_noise',
    ):

        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.name = name

        super(GaussianNoise, self).__init__()

    def __str__(self):
        additional_str = []

        if self._temp_data['is_train']:

            try:
                additional_str.append("mean: %s" % self.mean)
            except AttributeError:
                pass

            try:
                additional_str.append("stddev: %s" % self.stddev)
            except AttributeError:
                pass

            return self._str(additional_str)

        else:
            return self._skipped_layer_str()

    def build(self):

        if self._temp_data['is_train']:
            with tf.variable_scope(self.name):
                noise = tf.random_normal(
                    shape=self._temp_data['inputs'].get_shape(),
                    mean=self.mean,
                    stddev=self.stddev,
                    seed=self.seed,
                    dtype=self._temp_data['inputs'].dtype
                )
                self._temp_data['outputs'] = tf.add(self._temp_data['inputs'], noise)

        else:
            self._temp_data['outputs'] = self._temp_data['inputs']
