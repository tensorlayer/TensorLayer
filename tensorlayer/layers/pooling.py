# -*- coding: utf-8 -*-

import copy

from .core import *


class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer.
    You can choose ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D input or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D input.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer=None,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME',
            pool=tf.nn.max_pool,
            name='pool_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("PoolLayer   %s: ksize:%s strides:%s padding:%s pool:%s" % (self.name, str(ksize), str(strides), padding, pool.__name__))

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


def maxpool1d(net, filter_size=3, strides=2, padding='valid', data_format='channels_last', name=None):
    """Wrapper for `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d>`__ .

    Parameters
    ----------
    net : :class:`Layer`
        The previous layer with a output rank as 3.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 1-D layer with a output rank as 3.

    """
    logging.info("MaxPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.max_pooling1d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new


def meanpool1d(net, filter_size=3, strides=2, padding='valid', data_format='channels_last', name=None):
    """Wrapper for `tf.layers.average_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d>`__ .

    Parameters
    ------------
    net : :class:`Layer`
        The previous layer with a output rank as 3.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A mean pooling 1-D layer with a output rank as 3.

    """
    logging.info("MeanPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.average_pooling1d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new


def maxpool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : :class:`Layer`
        The previous layer with a output rank as 4.
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 2-D layer with a output rank as 4.

    """
    if strides is None:
        strides = filter_size
    assert len(strides) == 2, "len(strides) should be 2, MaxPool2d and PoolLayer are different."
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding, pool=tf.nn.max_pool, name=name)
    return net


def meanpool2d(net, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='meanpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : :class:`Layer`
        The previous layer with a output rank as 4.
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A mean pooling 2-D layer with a output rank as 4.

    """
    if strides is None:
        strides = filter_size
    assert len(strides) == 2, "len(strides) should be 2, MeanPool2d and PoolLayer are different."
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding, pool=tf.nn.avg_pool, name=name)
    return net


def maxpool3d(net, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='maxpool3d'):
    """Wrapper for `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d>`__ .

    Parameters
    ------------
    net : :class:`Layer`
        The previous layer with a output rank as 5.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 3-D layer with a output rank as 5.

    """
    logging.info("MaxPool3d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.max_pooling3d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new


def meanpool3d(net, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='meanpool3d'):
    """Wrapper for `tf.layers.average_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling3d>`__

    Parameters
    ------------
    net : :class:`Layer`
        The previous layer with a output rank as 5.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions must match the inputs.
        channels_last corresponds to inputs with the shape (batch, length, channels);
        while channels_first corresponds to inputs with shape (batch, channels, length).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A mean pooling 3-D layer with a output rank as 5.

    """
    logging.info("MeanPool3d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.average_pooling3d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new


# Alias
MaxPool1d = maxpool1d
MaxPool2d = maxpool2d
MaxPool3d = maxpool3d
MeanPool1d = meanpool1d
MeanPool2d = meanpool2d
MeanPool3d = meanpool3d
