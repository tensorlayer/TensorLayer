#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl


def get_variable_with_initializer(scope_name, var_name, shape, init=tl.initializers.random_normal()):
    var_name = scope_name + "/" + var_name
    initial_value = init(shape=shape)
    var = tf.Variable(initial_value=initial_value, name=var_name)
    return var

def conv2d(x, n_filter, filter_size, strides=(1, 1), padding='SAME',
           data_format='NHWC', dilation_rate=(1, 1),
           W_init=None,b_init=None, name='conv2d'):
    """
    2D convolution.

    Parameters
    ----------
    :param x:Tensor or variable.
    :param ksize: kernel shape
    :param strides:strides tupe
    :param padding:string , '"valid"' or '"same"'
    :param data_format:string, '"NHWC' or '"NCHW"'
    :param dilation_rate:tuple of 2 integers.
    :param W_init: initializer
        The initializer for the the weight matrix.
    :param b_init:initializer or None
        The initializer for the the bias vector. If None, skip biases.
    :param name:str ,name
    Result
    ----------
    :return:Tensor, result of 2D convolution

    Examples
    >>> net = tl.layers.Input([8, 400, 400, 3], name='input')
    >>> conv2d_1 = conv2d(net, n_filter=32, filter_size=(3, 3), strides=(2, 2), b_init=None, name='conv2d_1')
    >>> print(conv2d_1)
    """
    if W_init is None:
        W_init = tl.initializers.truncated_normal(stddev=0.02)
    else:
        W_init = W_init
    if data_format == 'NHWC':
        in_channel = x.get_shape()[-1]
    else:
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
    return x

def relu(x, alpha=0.):
    """
    relu activation
    Parameters
    ----------
    relu ,leaky_relu
    :param x: tensor or variable
    :param alpha:a scale ,slope of negative section
    Result
    :return:A tesor, relu activation
    """
    if alpha !=0.:
        x =  tf.nn.leaky_relu(x)
    else:
        x = tf.nn.relu(x)
    return x


# if __name__ == '__main__':
#     net = tl.layers.Input([8, 400, 400, 5], name='input')
#     b_init = tl.initializers.constant(value=0.1)
#     conv2d_1 = conv2d(net, n_filter=32, filter_size=(3, 3), strides=(2, 2), b_init=b_init, name='conv2d_1',data_format='NHWC')
#     conv2d_2 = conv2d(conv2d_1,n_filter=64, filter_size=(3, 3), strides=(2, 2), b_init=b_init, name='conv2d_2',data_format='NHWC')
#     print(conv2d_2)
