#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

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
        The mean. Default is 0.0.
    stddev : float
        The standard deviation. Default is 1.0.
    is_train : boolean
        Is trainable layer. If False, skip this layer. default is True.
    seed : int or None
        The seed for random noise.
    name : str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([64, 200], name='input')
    >>> net = tl.layers.Dense(n_units=100, act=tf.nn.relu, name='dense')(net)
    >>> gaussianlayer = tl.layers.GaussianNoise(name='gaussian')(net)
    >>> print(gaussianlayer)
    >>> output shape : (64, 100)

    """

    def __init__(
            self,
            mean=0.0,
            stddev=1.0,
            is_train=True,
            seed=None,
            name=None,  # 'gaussian_noise',
    ):
        super().__init__(name)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.is_train = is_train

        self.build()
        self._built = True

        logging.info("GaussianNoise %s: mean: %f stddev: %f" % (self.name, self.mean, self.stddev))

    def __repr__(self):
        s = '{classname}(mean={mean}, stddev={stddev}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs=None):
        pass

    def forward(self, inputs):
        if self.is_train is False:
            return inputs
        else:
            # noise = np.random.normal(0.0 , sigma , tf.to_int64(self.inputs).get_shape())
            noise = tf.random.normal(shape=inputs.get_shape(), mean=self.mean, stddev=self.stddev, seed=self.seed)
            outputs = inputs + noise
        return outputs
