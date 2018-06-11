#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            mean=0.0,
            stddev=1.0,
            is_train=True,
            seed=None,
            name='gaussian_noise_layer',
    ):
        super(GaussianNoiseLayer, self).__init__(prev_layer=prev_layer, name=name)

        if is_train is False:
            logging.info("  skip GaussianNoiseLayer")
            self.outputs = prev_layer.outputs

        else:
            logging.info("GaussianNoiseLayer %s: mean: %f stddev: %f" % (self.name, mean, stddev))
            with tf.variable_scope(name):
                # noise = np.random.normal(0.0 , sigma , tf.to_int64(self.inputs).get_shape())
                noise = tf.random_normal(shape=self.inputs.get_shape(), mean=mean, stddev=stddev, seed=seed)
                self.outputs = self.inputs + noise

            self._add_layers(self.outputs)
