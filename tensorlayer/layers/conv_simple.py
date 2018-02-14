#! /usr/bin/python
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
from .conv_pro import *

## Convolutional layer (Simplified)
def Conv1d(
        net,
        n_filter=32,
        filter_size=5,
        stride=1,
        dilation_rate=1,
        act=None,
        padding='SAME',
        use_cudnn_on_gpu=None,
        data_format="NWC",
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args={},
        b_init_args={},
        name='conv1d',
):
    """Wrapper for :class:`Conv1dLayer`, if you don't understand how to use :class:`Conv1dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : number of filter.
    filter_size : an int.
    stride : an int.
    dilation_rate : As it is 1D conv, the default is "NWC".
    act : None or activation function.
    others : see :class:`Conv1dLayer`.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, [batch_size, width])
    >>> y_ = tf.placeholder(tf.int64, shape=[batch_size,])
    >>> n = InputLayer(x, name='in')
    >>> n = ReshapeLayer(n, [-1, width, 1], name='rs')
    >>> n = Conv1d(n, 64, 3, 1, act=tf.nn.relu, name='c1')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m1')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c2')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m2')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c3')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m3')
    >>> n = FlattenLayer(n, name='f')
    >>> n = DenseLayer(n, 500, tf.nn.relu, name='d1')
    >>> n = DenseLayer(n, 100, tf.nn.relu, name='d2')
    >>> n = DenseLayer(n, 2, tf.identity, name='o')
    """
    if act is None:
        act = tf.identity
    net = Conv1dLayer(
        layer=net,
        act=act,
        shape=[filter_size, int(net.outputs.get_shape()[-1]), n_filter],
        stride=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        W_init=W_init,
        b_init=b_init,
        W_init_args=W_init_args,
        b_init_args=b_init_args,
        name=name,
    )
    return net


def Conv2d(
        net,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=None,
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args={},
        b_init_args={},
        use_cudnn_on_gpu=None,
        data_format=None,
        name='conv2d',
):
    """Wrapper for :class:`Conv2dLayer`, if you don't understand how to use :class:`Conv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : number of filter.
    filter_size : tuple (height, width) for filter size.
    strides : tuple (height, width) for strides.
    act : None or activation function.
    others : see :class:`Conv2dLayer`.

    Examples
    --------
    >>> w_init = tf.truncated_normal_initializer(stddev=0.01)
    >>> b_init = tf.constant_initializer(value=0.0)
    >>> inputs = InputLayer(x, name='inputs')
    >>> conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    >>> conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    >>> pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
    >>> conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    >>> conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    >>> pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
    """
    assert len(strides) == 2, "len(strides) should be 2, Conv2d and Conv2dLayer are different."
    if act is None:
        act = tf.identity

    try:
        pre_channel = int(net.outputs.get_shape()[-1])
    except:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        pre_channel = 1
        print("[warnings] unknow input channels, set to 1")
    net = Conv2dLayer(
        net,
        act=act,
        shape=[filter_size[0], filter_size[1], pre_channel, n_filter],  # 32 features for each 5x5 patch
        strides=[1, strides[0], strides[1], 1],
        padding=padding,
        W_init=W_init,
        W_init_args=W_init_args,
        b_init=b_init,
        b_init_args=b_init_args,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        name=name)
    return net


def DeConv2d(net,
             n_filter=32,
             filter_size=(3, 3),
             out_size=(30, 30),
             strides=(2, 2),
             padding='SAME',
             batch_size=None,
             act=None,
             W_init=tf.truncated_normal_initializer(stddev=0.02),
             b_init=tf.constant_initializer(value=0.0),
             W_init_args={},
             b_init_args={},
             name='decnn2d'):
    """Wrapper for :class:`DeConv2dLayer`, if you don't understand how to use :class:`DeConv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : int, number of output channel.
    filter_size : tuple of (height, width) for filter size.
    strides : tuple of (height, width) for strides.
    out_size : (require if TF version < 1.3) tuple of (height, width) of output (require if TF version < 1.3).
    batch_size : (require if TF version < 1.3) int or None, batch_size. If None, try to find the batch_size from the first dim of net.outputs (you should tell the batch_size when define the input placeholder).
    padding : 'VALID' or 'SAME'.
    act : None or activation function.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    name : A string
    """
    assert len(strides) == 2, "len(strides) should be 2, DeConv2d and DeConv2dLayer are different."
    if act is None:
        act = tf.identity

    if tf.__version__ > '1.3':
        print("  [TL] DeConv2d %s: n_filters:%s strides:%s pad:%s act:%s" % (
            name, str(n_filter), str(strides), padding, act.__name__))
        inputs = net.outputs
        scope_name = tf.get_variable_scope().name
        if scope_name:
            whole_name = scope_name + '/' + name
        else:
            whole_name = name
        net_new = Layer(inputs, name=whole_name)
        # with tf.name_scope(name):
        with tf.variable_scope(name) as vs:
            net_new.outputs = tf.contrib.layers.conv2d_transpose(inputs=inputs,
                            num_outputs=n_filter,
                            kernel_size=filter_size,
                            stride=strides,
                            padding=padding,
                            activation_fn=act,
                            weights_initializer=W_init,
                            biases_initializer=b_init,
                            scope=name)
            new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        net_new.all_layers = list(net.all_layers)
        net_new.all_params = list(net.all_params)
        net_new.all_drop = dict(net.all_drop)
        net_new.all_layers.extend([net_new.outputs])
        net_new.all_params.extend(new_variables)
        return net_new
    else:
        if batch_size is None:
            #     batch_size = tf.shape(net.outputs)[0]
            fixed_batch_size = net.outputs.get_shape().with_rank_at_least(1)[0]
            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                from tensorflow.python.ops import array_ops
                batch_size = array_ops.shape(net.outputs)[0]
        net = DeConv2dLayer(
            layer=net,
            act=act,
            shape=[filter_size[0], filter_size[1], n_filter, int(net.outputs.get_shape()[-1])],
            output_shape=[batch_size, int(out_size[0]), int(out_size[1]), n_filter],
            strides=[1, strides[0], strides[1], 1],
            padding=padding,
            W_init=W_init,
            b_init=b_init,
            W_init_args=W_init_args,
            b_init_args=b_init_args,
            name=name)
        return net


class DeConv3d(Layer):
    """
    The :class:`DeConv3d` class is a 3D transpose convolution layer, see `tf.contrib.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d_transpose>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_filter : Integer, the number of output filters.
    filter_size : A list of length 3 holding the [kernel_depth, kernel_height, kernel_width] of the filters. Can be an int if both values are the same.
    strides : A list of length 3: [stride_depth, stride_height, stride_width]. Can be an int if both strides are the same. Note that presently both strides must have the same value.
    padding : 'VALID' or 'SAME'.
    act : None or activation function.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    name : A string, an optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            n_filter=32,
            filter_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='SAME',
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            name='decnn3d'
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        if act is None:
            act = tf.identity

        print("  [TL] DeConv3d %s: n_filters:%s strides:%s pad:%s act:%s" % (
            name, str(n_filter), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            self.outputs = tf.contrib.layers.conv3d_transpose(
                        num_outputs=n_filter,
                        kernel_size=filter_size,
                        stride=strides,
                        padding=padding,
                        activation_fn=act,
                        weights_initializer=W_init,
                        biases_initializer=b_init,
                        scope=name,
                    )
            new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(new_variables)

class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`_.

    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, in_channels * channel_multiplier].

    Parameters
    ------------
    net : TensorLayer layer.
    channel_multiplier : int, The number of channels to expand to.
    filter_size : tuple (height, width) for filter size.
    strides : tuple (height, width) for strides.
    act : None or activation function.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    name : a string or None
        An optional name to attach to this layer.

    Examples
    ---------
    >>> t_im = tf.placeholder("float32", [None, 256, 256, 3])
    >>> net = InputLayer(t_im, name='in')
    >>> net = DepthwiseConv2d(net, 32, (3, 3), (1, 1, 1, 1), tf.nn.relu, padding="SAME", name='dep')
    >>> print(net.outputs.get_shape())
    ... (?, 256, 256, 96)

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`_
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`_
    """

    def __init__(
            self,
            layer=None,
            # n_filter = 32,
            channel_multiplier=3,
            shape=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            name='depthwise_conv2d',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        if act is None:
            act = tf.identity

        print("  [TL] DepthwiseConv2d %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))

        if act is None:
            act = tf.identity

        try:
            pre_channel = int(layer.outputs.get_shape()[-1])
        except:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            print("[warnings] unknow input channels, set to 1")

        shape = [shape[0], shape[1], pre_channel, channel_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        assert len(strides) == 4, "len(strides) should be 4."

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(
                name='W_sepconv2d', shape=shape, initializer=W_init, dtype=D_TYPE,
                **W_init_args)  # [filter_height, filter_width, in_channels, channel_multiplier]
            if b_init:
                b = tf.get_variable(name='b_sepconv2d', shape=(pre_channel * channel_multiplier), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding) + b)
            else:
                self.outputs = act(tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])
