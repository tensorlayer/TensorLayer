#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.keras import backend as tf_keras_backend


def get_variable_with_initializer(scope_name, var_name, shape, init=tl.initializers.random_normal()):
    var_name = scope_name + "/" + var_name
    initial_value = init(shape=shape)
    var = tf.Variable(initial_value=initial_value, name=var_name)
    return var

def preprocess_format(data_format, padding):
    if data_format in ['channels_last', 'NHWC'] :
        data_format = 'NHWC'
    elif data_format in ['channels_first', 'NCHW']:
        data_format = 'NCHW'
    else:
        raise Exception("Unsupported data format")
    if padding in ['SAME', 'same']:
        padding = 'SAME'
    elif padding in ['VALID', 'valid']:
        padding = 'VALID'
    else:
        raise Exception("Unsupported padding")
    return data_format, padding

def Input(shape, dtype=tf.float32, name=None):
    """
    The :class:`Input` class is the starting layer of a neural network.
    Parameters
    ----------
    :param shape: tuple (int), Including batch size.
    :param dtype: float, tf.float
    :param name: None or str, A unique layer name.
    Result
    :return:A tensor

    Examples
    ----------
    >>> x = Input([None,100,100,3], name='input')
    >>> print(x)
    """
    x = tf_keras_backend.placeholder(
        shape=shape, dtype=dtype, name=name)
    return x

def activation(x, act='relu'):
    """
    activations
    Parameters
    ----------
    relu ,leaky_relu
    :param x: tensor or variable
    :param alpha:a scale ,slope of negative section
    Result
    :return:Tesor, activation

    Examples
    ----------
    >>> x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
    >>> print(activation(x,'relu'))
    """
    _act_dict = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "leaky_relu": tf.nn.leaky_relu,
    "lrelu": tf.nn.leaky_relu,
    "softplus": tf.nn.softplus,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
    }
    if act not in _act_dict.keys():
        raise Exception("Unsupported act: {}".format(act))
    x = _act_dict[act](x)
    return x

def conv2d(x, n_filter, filter_size, strides=(1, 1), padding='same',
           data_format='channels_last', dilation_rate=(1, 1),
           W_init=None, b_init=None, act=None, name='conv2d'):
    """
    2D convolution.
    Parameters
    ----------
    :param x:Tensor or variable.
    :param filter_size: kernel shape
    :param strides:strides tupe
    :param padding:string , 'valid' or 'same'
    :param data_format:string, 'channels_last' or 'channels_first'
    :param dilation_rate:tuple of 2 integers.
    :param W_init: initializer
        The initializer for the the weight matrix.
    :param b_init:initializer or None
        The initializer for the the bias vector. If None, skip biases.
    :param act:string ,activation function
    :param name:string ,name
    Result
    ----------
    :return:Tensor, result of 2D convolution

    Examples
    ----------
    >>> net = tl.layers.Input([8, 400, 400, 3], name='input')
    >>> conv2d_1 = conv2d(net, n_filter=32, filter_size=(3, 3), strides=(2, 2), b_init=None, name='conv2d_1')
    >>> print(conv2d_1)
    """
    if W_init is None:
        W_init = tl.initializers.truncated_normal(stddev=0.02)
    else:
        W_init = W_init
    data_format, padding = preprocess_format(data_format, padding)
    if data_format == 'NHWC':
        in_channel = x.get_shape()[-1]
    elif data_format == 'NCHW':
        in_channel = x.get_shape()[1]
    ksize = (filter_size[0], filter_size[1], in_channel, n_filter)
    w_filter = get_variable_with_initializer(scope_name=name, var_name='filter', shape=ksize, init=W_init)
    x = tf.nn.conv2d(
        x,
        w_filter,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilation_rate,
        name=name
    )
    if b_init is not None:
        b_init = get_variable_with_initializer(scope_name=name, var_name='biases', shape=(n_filter,), init=b_init)
        x = tf.nn.bias_add(x, b_init,data_format=data_format,)
    if act is not None:
        x = activation(x, act)
    return x

def pool2d(x, filter_size, strides=(1,1),
           padding='same',data_format='channels_last',
           pool_mode='max', name='pool'):
    """
    Max pooling and Average pooling for 2D image.
    Parameters
    -----------
    :param x:Tensor or variable
    :param filter_size:kernel shape
    :param strides:strides tupe
    :param padding:string , '"valid"' or '"same"'
    :param data_format:string, '"channels_last' or '"channels_first"'
    :param pool_mode:string , '"max"' or '"avg"'
    :param name:srting, name
    Result
    ----------
    :return:Tensor

    Examples
    ----------
    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = pool2d(net, filter_size=(3, 3), strides=(2, 2), padding='same',pool_mode='max')
    >>> print(net)
    """
    data_format, padding = preprocess_format(data_format, padding)
    if pool_mode == 'max':
        x = tf.nn.max_pool(x, filter_size, strides,
                           padding=padding,
                           data_format=data_format,
                           name=name)
    elif pool_mode == 'avg':
        x = tf.nn.avg_pool(x, filter_size, strides,
                           padding=padding,
                           data_format=data_format,
                           name=name)
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))
    return x

def dense(x, n_units, act=None, W_init=None,
          b_init=None, name='dense'):
    """
    Parameters
    :param x:Tensor or variable
    :param n_units:int, The number of units of this layer.
    :param act:string ,activation function
    :param W_init:W_init: initializer
        The initializer for the the weight matrix.
    :param b_init:initializer or None
        The initializer for the the bias vector. If None, skip biases.
    :param name:string, name
    :return:A tensor

    Examples
    >>> net = tl.layers.Input([None,10], name='input')
    >>> net = dense(net, 50, act='relu', name='dense')
    >>> print(net)
    """
    if W_init is None:
        W_init = tl.initializers.truncated_normal(stddev=0.05)
    else:
        W_init = W_init
    ksize = (x.get_shape()[-1], n_units)
    w_filter = get_variable_with_initializer(scope_name=name, var_name='filter', shape=ksize, init=W_init)
    x= tf.matmul(x, w_filter)
    if b_init is not None:
        b_init = get_variable_with_initializer(scope_name=name, var_name='biases', shape=(n_units,), init=b_init)
        x = tf.add(x, b_init)
    if act is not None:
        x = activation(x, act)
    return x

def flatten(x):
    """Flatten a tensor.

    Parameters
    :param x: A tensor or variable.
    Result
    ----------
    :return:A tensor, reshaped into 1-D
    """
    return tf.reshape(x, [-1])

if __name__ == '__main__':
    # input
    # a = Input([None,100,100,3], name='input')
    # print(a)

    # convlutions
    # net = tl.layers.Input([8, 400, 400, 5], name='input')
    # b_init = tl.initializers.constant(value=0.1)
    # conv2d_1 = conv2d(net, n_filter=32, filter_size=(3, 3), strides=(2, 2), b_init=b_init, name='conv2d_1',act='lrelu',padding='valid', data_format='NHWC')
    # conv2d_2 = conv2d(conv2d_1,n_filter=64, filter_size=(3, 3), strides=(2, 2), b_init=b_init, name='conv2d_2',act='softplus')
    # print(conv2d_2)

    # 'activation'
    # x = tf.constant([-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")])
    # print(_activation(x,'relu'))

    # pooling
    # net = tl.layers.Input([3, 50, 50, 32], name='input')
    # net = pool2d(net, filter_size=(5, 5), strides=(1, 1), padding='same', pool_mode='avg')
    # print(net)

    # flatten
    # net = tl.layers.Input([3, 50, 50, 32], name='input')
    # print(flatten(net))

    # dense
    net = tl.layers.Input([None,10], name='input')
    net = dense(net, 50, act='relu', name='dense')
    print(net)