#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf


_dtypeDict = {
    'DType': tf.DType,
    'float16': tf.float16,
    'float32': tf.float32,
    'float64': tf.float64,
    'int8': tf.int8,
    'int16': tf.int16,
    'int32': tf.int32,
    'int64': tf.int64,
    'uint8': tf.uint8,
    'uint16': tf.uint16,
    'uint32': tf.uint32,
    'uint64': tf.uint64
}


def Variable(initial_value, name):
    var = tf.Variable(initial_value=initial_value, name=name)
    return var


def matmul(a, b):
    outputs = tf.matmul(a, b)
    return outputs


def add(value, bias, name=None):
    outputs = tf.add(value, bias, name=name)
    return outputs


def dtypes(dt):
    """

    Parameters
    ----------
    dt : string
         It could be 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'DType'.
    Returns
    -------

    """
    if dt not in _dtypeDict.keys():
        raise Exception("Unsupported act: {}".format(dt))
    return _dtypeDict[dt]


def minimum(x, y, name=None):
    outputs = tf.minimum(
        x=x,
        y=y,
        name=name
    )
    return outputs


def reshape(tensor, shape):
    return tf.reshape(tensor, shape)


def concat(values, axis):
    return tf.concat(values, axis)


def convert_to_tensor(value, dtype=None):
    return tf.convert_to_tensor(value, dtype)


def sqrt(x):
    return tf.sqrt(x)


def reduce_mean(input_tensor, axis=None, name=None):
    return tf.reduce_mean(input_tensor, axis=axis, name=name)


def reduce_max(input_tensor, axis=None, name=None):
    return tf.reduce_max(input_tensor, axis=axis, name=name)


def reduce_min(input_tensor, axis=None, name=None):
    return tf.reduce_min(input_tensor, axis=axis, name=name)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0, name=None):
    outputs = tf.pad(
        tensor,
        paddings,
        mode=mode,
        constant_values=constant_values,
        name=name)
    return outputs


def constant(value, dtype=None):
    return tf.constant(value, dtype=dtype)


def stack(values, axis=0):
    return tf.stack(values, axis=axis)


def meshgrid(x, y):
    return tf.meshgrid(x, y)


def range(start, limit=None, delta=1, dtype=None):
    if limit is None:
        outputs = tf.range(start, delta=delta, dtype=dtype)
    else:
        outputs = tf.range(start, limit, delta=delta, dtype=dtype)
    return outputs


def expand_dims(input, axis):
    return tf.expand_dims(input, axis)


def tile(input, multiples):
    return tf.tile(input, multiples)


def cast(x, dtype):
    return tf.cast(x, dtype=dtype)


def transpose(a, perm=None, conjugate=False):
    return tf.transpose(a, perm, conjugate)


def gather_nd(params, indices, batch_dims=0):
    return tf.gather_nd(params, indices, batch_dims)


def clip_by_value(t, clip_value_min, clip_value_max):
    return tf.clip_by_value(t, clip_value_min, clip_value_max)


def split(value, num_or_size_splits, axis=0, num=None):
    return tf.split(value=value, num_or_size_splits=num_or_size_splits, axis=axis, num=num)


if __name__ == '__main__':
    x = tf.Variable(tf.random.uniform([5, 30], -1, 1))
    s0, s1, s2 = tf.split(x, num_or_size_splits=3, axis=1)
    print(s0,s1,s2)
    print(split(x,3,1))
