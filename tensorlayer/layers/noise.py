#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
    >>> net = tl.layers.Input(x, name='input')
    >>> net = tl.layers.Dense(net, n_units=100, act=tf.nn.relu, name='dense3')
    >>> net = tl.layers.GaussianNoise(net, name='gaussian')
    (64, 100)

    """

    def __init__(
            self,
            # prev_layer,
            mean=0.0,
            stddev=1.0,
            seed=None,
            name=None,  #'gaussian_noise',
    ):
        # super(GaussianNoise, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        logging.info("GaussianNoise %s: mean: %f stddev: %f" % (self.name, self.mean, self.stddev))

    def build(self, inputs):
        pass

    def forward(self, inputs, train):
        if train is False:
            outputs = inputs
        else:
            # noise = np.random.normal(0.0 , sigma , tf.to_int64(self.inputs).get_shape())
            noise = tf.random.normal(shape=inputs.get_shape(), mean=self.mean, stddev=self.stddev, seed=self.seed)
            outputs = inputs + noise
        return outputs
