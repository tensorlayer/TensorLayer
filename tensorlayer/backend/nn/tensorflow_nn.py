#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def padding_format(padding):
    if padding in ["SAME", "same"]:
        padding = "SAME"
    elif padding in ["VALID", "valid"]:
        padding = "VALID"
    else:
        raise Exception("Unsupported padding: " + str(padding))
    return padding


def preprocess_1d_format(data_format, padding):
    if data_format in ["channels_last", "NWC"]:
        data_format = "NWC"
    elif data_format in ["channels_first", "NCW"]:
        data_format = "NCW"
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def preprocess_2d_format(data_format, padding):
    if data_format in ["channels_last", "NHWC"]:
        data_format = "NHWC"
    elif data_format in ["channels_first", "NCHW"]:
        data_format = "NCHW"
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def preprocess_3d_format(data_format, padding):
    if data_format in ['channels_last', 'NDHWC']:
        data_format = 'NDHWC'
    elif data_format in ['channels_first', 'NCDHW']:
        data_format = 'NCDHW'
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def nchw_to_nhwc(x):
    if len(x.shape) == 3:
        x = tf.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = tf.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 5:
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    else:
        raise Exception("Unsupported dimensions")
    return x


def nhwc_to_nchw(x):
    if len(x.shape) == 3:
        x = tf.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = tf.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 5:
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    else:
        raise Exception("Unsupported dimensions")
    return x


def relu(x):
    return tf.nn.relu(x)


def relu6(x):
    return tf.nn.relu6(x)


def leaky_relu(x):
    return tf.nn.leaky_relu(x)


def softplus(x):
    return tf.nn.softplus(x)


def tanh(x):
    return tf.nn.tanh(x)


def sigmoid(x):
    return tf.nn.sigmoid(x)


def softmax(logits, axis=None):
    return tf.nn.softmax(logits, axis)


def bias_add(x, bias, data_format=None, name=None):
    x = tf.nn.bias_add(x, bias, data_format=data_format, name=name)
    return x


def conv1d(input, filters, stride, padding, data_format='NWC',
           dilations=None, name=None):
    data_format, padding = preprocess_1d_format(data_format, padding)
    outputs = tf.nn.conv1d(
        input=input,
        filters=filters,
        strides=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )
    return outputs


def conv2d(input, filters, strides, padding, data_format='NHWC',
           dilations=None, name=None):
    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.conv2d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )
    return outputs


def conv3d(input, filters, strides, padding, data_format='NDHWC',
           dilations=None, name=None):
    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,  # 'NDHWC',
        dilations=dilations,  # [1, 1, 1, 1, 1],
        name=name,
    )
    return outputs


def lrn(inputs, depth_radius, bias, alpha, beta):
    outputs = tf.nn.lrn(inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
    return outputs


def moments(x, axes, shift=None, keepdims=False):
    outputs = tf.nn.moments(x, axes, shift, keepdims)
    return outputs


def max_pool(input, ksize, strides, padding, name=None):
    padding = padding_format(padding)
    outputs = tf.nn.max_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        name=name
    )
    return outputs


def avg_pool(input, ksize, strides, padding, name=None):
    padding = padding_format(padding)
    outputs = tf.nn.avg_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        name=name
    )
    return outputs


def max_pool3d(input, ksize, strides, padding, data_format=None, name=None):
    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.max_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name
    )
    return outputs


def avg_pool3d(input, ksize, strides, padding, data_format=None, name=None):
    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.avg_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )
    return outputs


def pool(input, window_shape, pooling_type, strides=None, padding='VALID',
         data_format=None, dilations=None, name=None):
    if pooling_type in ["MAX", "max"]:
        pooling_type = "MAX"
    elif pooling_type in ["AVG", "avg"]:
        pooling_type = "AVG"
    else:
        raise ValueError('Unsupported pool_mode: ' + str(pooling_type))
    padding = padding_format(padding)
    outputs = tf.nn.pool(
        input=input,
        window_shape=window_shape,
        pooling_type=pooling_type,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


def depthwise_conv2d(input, filter, strides, padding,
                     data_format=None, dilations=None, name=None):
    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.depthwise_conv2d(
        input=input,
        filter=filter,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


def conv1d_transpose(input, filters, output_shape, strides, padding='SAME',
                     data_format='NWC', dilations=None, name=None):
    data_format, padding = preprocess_1d_format(data_format, padding)
    outputs = tf.nn.conv1d_transpose(
        input=input,
        filters=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


def conv2d_transpose(input, filters, output_shape, strides, padding='SAME',
                     data_format='NHWC', dilations=None, name=None):
    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.conv2d_transpose(
        input=input,
        filters=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )


def conv3d_transpose(input, filters, output_shape, strides, padding='SAME',
                     data_format='NDHWC', dilations=None, name=None):
    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d_transpose(
        input=input,
        filters=filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name
    )
    return outputs



