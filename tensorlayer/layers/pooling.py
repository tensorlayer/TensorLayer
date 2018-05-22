# -*- coding: utf-8 -*-

import copy

import tensorflow as tf

from .. import _logging as logging
from .core import *

from ..deprecation import deprecated_alias

__all__ = [
    'PoolLayer',
    'MaxPool1d',
    'MeanPool1d',
    'MaxPool2d',
    'MeanPool2d',
    'MaxPool3d',
    'MeanPool3d',
    'GlobalMaxPool1d',
    'GlobalMeanPool1d',
    'GlobalMaxPool2d',
    'GlobalMeanPool2d',
    'GlobalMaxPool3d',
    'GlobalMeanPool3d',
]


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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
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
            "PoolLayer   %s: ksize:%s strides:%s padding:%s pool:%s" %
            (name, str(ksize), str(strides), padding, pool.__name__)
        )

        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class MaxPool1d(Layer):
    """Max pooling for 1D signal [batch, length, channel]. Wrapper for `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
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

    """

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=3, strides=2, padding='valid', data_format='channels_last', name='maxpool1d'
    ):
        super(MaxPool1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MaxPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding))
        )
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.layers.max_pooling1d(
            self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )
        # update layer (customized)
        self.all_layers.append(self.outputs)


class MeanPool1d(Layer):
    """Mean pooling for 1D signal [batch, length, channel]. Wrapper for `tf.layers.average_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d>`__ .

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
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

    """
    # logging.info("MeanPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    # outputs = tf.layers.average_pooling1d(prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)
    #
    # net_new = copy.copy(prev_layer)
    # net_new.outputs = outputs
    # net_new.all_layers.extend([outputs])
    # return net_new
    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=3, strides=2, padding='valid', data_format='channels_last', name='meanpool1d'
    ):
        super(MeanPool1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MeanPool1d %s: filter_size:%s strides:%s padding:%s" %
            (name, str(filter_size), str(strides), str(padding))
        )

        # operation (customized)
        self.outputs = tf.layers.average_pooling1d(
            prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        # update layer (customized)
        self.all_layers.append(self.outputs)


class MaxPool2d(Layer):
    """Max pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    name : str
        A unique layer name.

    """

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool2d'):
        if strides is None:
            strides = filter_size

        super(MaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MaxPool2d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding))
        )
        self.inputs = prev_layer.outputs
        # operation (customized)
        if tf.__version__ > '1.5':
            self.outputs = tf.layers.max_pooling2d(
                self.inputs, filter_size, strides, padding=padding, data_format='channels_last', name=name
            )
        else:
            if len(strides) != 2:
                raise Exception("len(strides) should be 2.")
            ksize = [1, filter_size[0], filter_size[1], 1]
            strides = [1, strides[0], strides[1], 1]
            self.outputs = tf.nn.max_pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class MeanPool2d(Layer):
    """Mean pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    name : str
        A unique layer name.

    """

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='meanpool2d'):
        if strides is None:
            strides = filter_size

        super(MeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MeanPool2d %s: filter_size:%s strides:%s padding:%s" %
            (name, str(filter_size), str(strides), str(padding))
        )
        self.inputs = prev_layer.outputs
        # operation (customized)
        if tf.__version__ > '1.5':
            self.outputs = tf.layers.average_pooling2d(
                self.inputs, filter_size, strides, padding=padding, data_format='channels_last', name=name
            )
        else:
            if len(strides) != 2:
                raise Exception("len(strides) should be 2.")
            ksize = [1, filter_size[0], filter_size[1], 1]
            strides = [1, strides[0], strides[1], 1]
            self.outputs = tf.nn.avg_pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


# def maxpool3d(net, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='maxpool3d'):
class MaxPool3d(Layer):
    """Max pooling for 3D volume [batch, depth, height, width, channel]. Wrapper for `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d>`__ .

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last',
            name='maxpool3d'
    ):
        super(MaxPool3d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MaxPool3d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding))
        )
        # operation (customized)
        self.inputs = prev_layer.outputs
        self.outputs = tf.layers.max_pooling3d(
            self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        # update layer (customized)
        self.all_layers.append(self.outputs)


# def meanpool3d(net, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last', name='meanpool3d'):
class MeanPool3d(Layer):
    """Mean pooling for 3D volume [batch, depth, height, width, channel]. Wrapper for `tf.layers.average_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling3d>`__

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last',
            name='meanpool3d'
    ):

        super(MeanPool3d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "MeanPool3d %s: filter_size:%s strides:%s padding:%s" %
            (name, str(filter_size), str(strides), str(padding))
        )

        self.inputs = prev_layer.outputs

        # operation (customized)
        self.outputs = tf.layers.average_pooling3d(
            prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMaxPool1d(Layer):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = InputLayer(x, name='in')
    >>> n = GlobalMaxPool1d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmaxpool1d'):
        super(GlobalMaxPool1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("GlobalMaxPool1d %s" % name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_max(self.inputs, axis=1, name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMeanPool1d(Layer):
    """The :class:`GlobalMeanPool1d` class is a 1D Global Mean Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = InputLayer(x, name='in')
    >>> n = GlobalMeanPool1d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmeanpool1d'):
        super(GlobalMeanPool1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("GlobalMeanPool1d %s" % name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_mean(self.inputs, axis=1, name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMaxPool2d(Layer):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = InputLayer(x, name='in2')
    >>> n = GlobalMaxPool2d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmaxpool2d'):
        super(GlobalMaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("GlobalMaxPool2d %s" % name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_max(self.inputs, axis=[1, 2], name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMeanPool2d(Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = InputLayer(x, name='in2')
    >>> n = GlobalMeanPool2d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmeanpool2d'):
        super(GlobalMeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("GlobalMeanPool2d %s" % name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_mean(self.inputs, axis=[1, 2], name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMaxPool3d(Layer):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 100, 100, 30])
    >>> n = InputLayer(x, name='in')
    >>> n = GlobalMaxPool3d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmaxpool3d'):
        super(GlobalMaxPool3d, self).__init__(prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        # print out info (customized)
        logging.info("GlobalMaxPool3d %s" % name)
        # operation (customized)
        self.outputs = tf.reduce_max(self.inputs, axis=[1, 2, 3], name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


class GlobalMeanPool3d(Layer):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 100, 100, 30])
    >>> n = InputLayer(x, name='in')
    >>> n = GlobalMeanPool2d(n)
    ... [None, 30]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, name='globalmeanpool3d'):
        super(GlobalMeanPool3d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("GlobalMeanPool3d %s" % name)
        self.inputs = prev_layer.outputs
        # operation (customized)
        self.outputs = tf.reduce_mean(self.inputs, axis=[1, 2, 3], name=name)
        # update layer (customized)
        self.all_layers.append(self.outputs)


# Alias
# MaxPool1d = maxpool1d
# MaxPool2d = maxpool2d
# MeanPool1d = meanpool1d
# MeanPool2d = meanpool2d
