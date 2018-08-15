#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

__all__ = [
    'calculate_output_shape'
]


def calculate_output_shape(input, filter_size_h, filter_size_w,
    stride_h, stride_w, num_outputs, padding='SAME', data_format='NHWC'):

    #calculation of the output_shape:
    if data_format == "NHWC":
        input_channel_size = input.get_shape().as_list()[3]
        input_size_h = input.get_shape().as_list()[1]
        input_size_w = input.get_shape().as_list()[2]
        stride_shape = [1, stride_h, stride_w, 1]

        if padding == 'VALID':
            output_size_h = (input_size_h - 1)*stride_h + filter_size_h
            output_size_w = (input_size_w - 1)*stride_w + filter_size_w

        elif padding == 'SAME':
            output_size_h = (input_size_h - 1)*stride_h + 1
            output_size_w = (input_size_w - 1)*stride_w + 1
        else:
            raise ValueError("unknown padding")

        output_shape = [tf.shape(input)[0], output_size_h, output_size_w, num_outputs]

    elif data_format == "NCHW":
        input_channel_size = input.get_shape().as_list()[1]
        input_size_h = input.get_shape().as_list()[2]
        input_size_w = input.get_shape().as_list()[3]
        stride_shape = [1, 1, stride_h, stride_w]

        if padding == 'VALID':
            output_size_h = (input_size_h - 1)*stride_h + filter_size_h
            output_size_w = (input_size_w - 1)*stride_w + filter_size_w

        elif padding == 'SAME':
            output_size_h = (input_size_h - 1)*stride_h + 1
            output_size_w = (input_size_w - 1)*stride_w + 1

        else:
            raise ValueError("unknown padding")

        output_shape =[tf.shape(input)[0], output_size_h, output_size_w, num_outputs]

    else:
        raise ValueError("unknown data_format")

    return output_shape
