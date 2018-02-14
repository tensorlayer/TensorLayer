# -*- coding: utf-8 -*-

import copy
import inspect
import random
import time
import warnings

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer, you can choose
    ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    ksize : a list of ints that has length >= 4.
        The size of the window for each dimension of the input tensor.
    strides : a list of ints that has length >= 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    pool : a pooling function
        - see `TensorFlow pooling APIs <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`_
        - class ``tf.nn.max_pool``
        - class ``tf.nn.avg_pool``
        - class ``tf.nn.max_pool3d``
        - class ``tf.nn.avg_pool3d``
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - see :class:`Conv2dLayer`.
    """

    def __init__(
            self,
            layer=None,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            pool=tf.nn.max_pool,
            name='pool_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] PoolLayer   %s: ksize:%s strides:%s padding:%s pool:%s" % (self.name, str(ksize), str(strides), padding, pool.__name__))

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


def create_maxpool1d(net, filter_size, strides, padding='valid', data_format='channels_last', name=None):  #Untested
    """Wrapper for `tf.layers.max_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling1d>`_ .

    Parameters
    ------------
    net : TensorLayer layer, the tensor over which to pool. Must have rank 3.
    filter_size (pool_size) : An integer or tuple/list of a single integer, representing the size of the pooling window.
    strides : An integer or tuple/list of a single integer, specifying the strides of the pooling operation.
    padding : A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    data_format : A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, length, channels) while channels_first corresponds to inputs with shape (batch, channels, length).
    name : A string, the name of the layer.

    Returns
    --------
    - A :class:`Layer` which the output tensor, of rank 3.
    """
    print("  [TL] MaxPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.max_pooling1d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new

MaxPool1d = create_maxpool1d

def create_meanpool1d(net, filter_size, strides, padding='valid', data_format='channels_last', name=None):  #Untested
    """Wrapper for `tf.layers.average_pooling1d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling1d>`_ .

    Parameters
    ------------
    net : TensorLayer layer, the tensor over which to pool. Must have rank 3.
    filter_size (pool_size) : An integer or tuple/list of a single integer, representing the size of the pooling window.
    strides : An integer or tuple/list of a single integer, specifying the strides of the pooling operation.
    padding : A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    data_format : A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, length, channels) while channels_first corresponds to inputs with shape (batch, channels, length).
    name : A string, the name of the layer.

    Returns
    --------
    - A :class:`Layer` which the output tensor, of rank 3.
    """
    print("  [TL] MeanPool1d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.average_pooling1d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new

MeanPool1d = create_meanpool1d

def create_maxpool2d(net, filter_size=(2, 2), strides=None, padding='SAME', name='maxpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : TensorLayer layer.
    filter_size : tuple of (height, width) for filter size.
    strides : tuple of (height, width). Default is the same with filter_size.
    others : see :class:`PoolLayer`.
    """
    if strides is None:
        strides = filter_size
    assert len(strides) == 2, "len(strides) should be 2, MaxPool2d and PoolLayer are different."
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding, pool=tf.nn.max_pool, name=name)
    return net

MaxPool2d = create_maxpool2d

def create_meanpool2d(net, filter_size=(2, 2), strides=None, padding='SAME', name='meanpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : TensorLayer layer.
    filter_size : tuple of (height, width) for filter size.
    strides : tuple of (height, width). Default is the same with filter_size.
    others : see :class:`PoolLayer`.
    """
    if strides is None:
        strides = filter_size
    assert len(strides) == 2, "len(strides) should be 2, MeanPool2d and PoolLayer are different."
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1], strides=[1, strides[0], strides[1], 1], padding=padding, pool=tf.nn.avg_pool, name=name)
    return net

MeanPool2d = create_meanpool2d

def create_maxpool3d(net, filter_size, strides, padding='valid', data_format='channels_last', name=None):  #Untested
    """Wrapper for `tf.layers.max_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling3d>`_ .

    Parameters
    ------------
    net : TensorLayer layer, the tensor over which to pool. Must have rank 5.
    filter_size (pool_size) : An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width) specifying the size of the pooling window. Can be a single integer to specify the same value for all spatial dimensions.
    strides : An integer or tuple/list of 3 integers, specifying the strides of the pooling operation. Can be a single integer to specify the same value for all spatial dimensions.
    padding : A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    data_format : A string. The ordering of the dimensions in the inputs. channels_last (default) and channels_first are supported. channels_last corresponds to inputs with shape (batch, depth, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, depth, height, width).
    name : A string, the name of the layer.
    """
    print("  [TL] MaxPool3d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.max_pooling3d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new

MaxPool3d = create_maxpool3d

def create_meanpool3d(net, filter_size, strides, padding='valid', data_format='channels_last', name=None):  #Untested
    """Wrapper for `tf.layers.average_pooling3d <https://www.tensorflow.org/api_docs/python/tf/layers/average_pooling3d>`_

    Parameters
    ------------
    net : TensorLayer layer, the tensor over which to pool. Must have rank 5.
    filter_size (pool_size) : An integer or tuple/list of 3 integers: (pool_depth, pool_height, pool_width) specifying the size of the pooling window. Can be a single integer to specify the same value for all spatial dimensions.
    strides : An integer or tuple/list of 3 integers, specifying the strides of the pooling operation. Can be a single integer to specify the same value for all spatial dimensions.
    padding : A string. The padding method, either 'valid' or 'same'. Case-insensitive.
    data_format : A string. The ordering of the dimensions in the inputs. channels_last (default) and channels_first are supported. channels_last corresponds to inputs with shape (batch, depth, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, depth, height, width).
    name : A string, the name of the layer.
    """
    print("  [TL] MeanPool3d %s: filter_size:%s strides:%s padding:%s" % (name, str(filter_size), str(strides), str(padding)))
    outputs = tf.layers.average_pooling3d(net.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)

    net_new = copy.copy(net)
    net_new.outputs = outputs
    net_new.all_layers.extend([outputs])
    return net_new

MeanPool3d = create_meanpool3d
