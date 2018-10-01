#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = ['compute_deconv2d_output_shape', 'compute_deconv3d_output_shape']


def compute_deconv2d_output_shape(
    input, filter_size_h, filter_size_w, stride_h, stride_w, num_outputs, padding='SAME', data_format='NHWC'
):
    try:
        batch_size = int(input.get_shape()[0])
    except:
        batch_size = tf.shape(input)[0]

    #calculation of the output_shape:
    if data_format == "NHWC":
        input_channel_size = input.get_shape().as_list()[3]
        input_size_h = input.get_shape().as_list()[1]
        input_size_w = input.get_shape().as_list()[2]
        stride_shape = [1, stride_h, stride_w, 1]

    elif data_format == "NCHW":
        input_channel_size = input.get_shape().as_list()[1]
        input_size_h = input.get_shape().as_list()[2]
        input_size_w = input.get_shape().as_list()[3]
        stride_shape = [1, 1, stride_h, stride_w]

    else:
        raise ValueError("unknown data_format")

    if padding == 'VALID':
        output_size_h = (input_size_h - 1) * stride_h + filter_size_h
        output_size_w = (input_size_w - 1) * stride_w + filter_size_w

    elif padding == 'SAME':
        output_size_h = (input_size_h - 1) * stride_h + 1
        output_size_w = (input_size_w - 1) * stride_w + 1
    else:
        raise ValueError("unknown padding")

    if data_format == "NHWC":
        return [batch_size, output_size_h, output_size_w, num_outputs]

    else:  # data_format == "NCHW"
        return [batch_size, num_outputs, output_size_h, output_size_w]


def compute_deconv3d_output_shape(
    input,
    filter_size_d,
    filter_size_h,
    filter_size_w,
    stride_d,
    stride_h,
    stride_w,
    num_outputs,
    padding='SAME',
    data_format='NDHWC'
):
    try:
        batch_size = int(input.get_shape()[0])
    except:
        batch_size = tf.shape(input)[0]

    if data_format == "NDHWC":
        input_channel_size = input.get_shape().as_list()[3]
        input_size_d = input.get_shape().as_list()[1]
        input_size_h = input.get_shape().as_list()[2]
        input_size_w = input.get_shape().as_list()[3]
        stride_shape = [1, stride_h, stride_w, 1]

    elif data_format == "NCDHW":
        input_channel_size = input.get_shape().as_list()[1]
        input_size_d = input.get_shape().as_list()[2]
        input_size_h = input.get_shape().as_list()[3]
        input_size_w = input.get_shape().as_list()[4]
        stride_shape = [1, 1, stride_h, stride_w]

    else:
        raise ValueError("unknown data_format")

    if padding == 'VALID':
        output_size_d = (input_size_d - 1) * stride_d + filter_size_d
        output_size_h = (input_size_h - 1) * stride_h + filter_size_h
        output_size_w = (input_size_w - 1) * stride_w + filter_size_w

    elif padding == 'SAME':
        output_size_d = (input_size_d - 1) * stride_d + 1
        output_size_h = (input_size_h - 1) * stride_h + 1
        output_size_w = (input_size_w - 1) * stride_w + 1

    else:
        raise ValueError("unknown padding")

    if data_format == "NDHWC":
        return [batch_size, output_size_d, output_size_h, output_size_w, num_outputs]

    else:  # data_format == "NCDHW"
        return [batch_size, num_outputs, output_size_d, output_size_h, output_size_w]
