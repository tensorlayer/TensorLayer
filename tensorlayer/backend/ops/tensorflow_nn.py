#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import moving_averages
from math import floor, ceil
# loss function
sparse_softmax_cross_entropy_with_logits = tf.nn.sparse_softmax_cross_entropy_with_logits
sigmoid_cross_entropy_with_logits = tf.nn.sigmoid_cross_entropy_with_logits


def padding_format(padding):
    """
    Checks that the padding format correspond format.

    Parameters
    ----------
    padding : str
        Must be one of the following:"same", "SAME", "VALID", "valid"

    Returns
    -------
        str "SAME" or "VALID"
    """

    if padding in ["SAME", "same"]:
        padding = "SAME"
    elif padding in ["VALID", "valid"]:
        padding = "VALID"
    elif padding == None:
        padding = None
    else:
        raise Exception("Unsupported padding: " + str(padding))
    return padding


def preprocess_1d_format(data_format, padding):
    """
    Checks that the 1-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NWC","NCW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NWC" or "NCW" and "SAME" or "VALID"
    """
    if data_format in ["channels_last", "NWC"]:
        data_format = "NWC"
    elif data_format in ["channels_first", "NCW"]:
        data_format = "NCW"
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def preprocess_2d_format(data_format, padding):
    """
    Checks that the 2-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NHWC","NCHW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NHWC" or "NCHW" and "SAME" or "VALID"
    """

    if data_format in ["channels_last", "NHWC"]:
        data_format = "NHWC"
    elif data_format in ["channels_first", "NCHW"]:
        data_format = "NCHW"
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def preprocess_3d_format(data_format, padding):
    """
    Checks that the 3-D dataformat format correspond format.

    Parameters
    ----------
    data_format : str
        Must be one of the following:"channels_last","NDHWC","NCDHW","channels_first"
    padding : str
        Must be one of the following:"same","valid","SAME","VALID"

    Returns
    -------
        str "NDHWC" or "NCDHW" and "SAME" or "VALID"
    """

    if data_format in ['channels_last', 'NDHWC']:
        data_format = 'NDHWC'
    elif data_format in ['channels_first', 'NCDHW']:
        data_format = 'NCDHW'
    elif data_format == None:
        data_format = None
    else:
        raise Exception("Unsupported data format: " + str(data_format))
    padding = padding_format(padding)
    return data_format, padding


def nchw_to_nhwc(x):
    """
    Channels first to channels last

    Parameters
    ----------
    x : tensor
        channels first tensor data

    Returns
    -------
        channels last tensor data
    """

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
    """
    Channles last to channels first

    Parameters
    ----------
    x : tensor
        channels last tensor data

    Returns
    -------
        channels first tensor data
    """

    if len(x.shape) == 3:
        x = tf.transpose(x, (0, 2, 1))
    elif len(x.shape) == 4:
        x = tf.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 5:
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    else:
        raise Exception("Unsupported dimensions")
    return x


class ReLU(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.relu(x)


def relu(x):
    """
    Computes rectified linear: max(features, 0).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8, int16,
        int8, int64, bfloat16, uint16, half, uint32, uint64, qint8.

    Returns
    -------
        A Tensor. Has the same type as features.
    """

    return tf.nn.relu(x)


class ReLU6(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.relu6(x)


def relu6(x):
    """
    Computes Rectified Linear 6: min(max(features, 0), 6).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: float32, float64, int32, uint8, int16,
        int8, int64, bfloat16, uint16, half, uint32, uint64, qint8.

    Returns
    -------
        A Tensor with the same type as features.
    """

    return tf.nn.relu6(x)


class LeakyReLU(object):

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return tf.nn.leaky_relu(x, alpha=self.alpha)


def leaky_relu(x, alpha=0.2):
    """
    Compute the Leaky ReLU activation function.

    Parameters
    ----------
    x : tensor
        representing preactivation values. Must be one of the following types:
        float16, float32, float64, int32, int64.

    Returns
    -------
        The activation value.
    """

    return tf.nn.leaky_relu(x, alpha=alpha)


class Softplus(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.softplus(x)


def softplus(x):
    """
    Computes softplus: log(exp(features) + 1).

    Parameters
    ----------
    x : tensor
        Must be one of the following types: half, bfloat16, float32, float64.

    Returns
    -------
        A Tensor. Has the same type as features.
    """

    return tf.nn.softplus(x)


class Tanh(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.tanh(x)


def tanh(x):
    """
    Computes hyperbolic tangent of x element-wise.

    Parameters
    ----------
    x : tensor
        Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.
    """

    return tf.nn.tanh(x)


class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.sigmoid(x)


def sigmoid(x):
    """
    Computes sigmoid of x element-wise.

    Parameters
    ----------
    x : tensor
        A Tensor with type float16, float32, float64, complex64, or complex128.

    Returns
    -------
        A Tensor with the same type as x.
    """

    return tf.nn.sigmoid(x)


class Softmax(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.nn.softmax(x)


def softmax(logits, axis=None):
    """
    Computes softmax activations.

    Parameters
    ----------
    logits : tensor
        Must be one of the following types: half, float32, float64.
    axis : int
        The dimension softmax would be performed on. The default is -1 which indicates the last dimension.

    Returns
    -------
        A Tensor. Has the same type and shape as logits.
    """

    return tf.nn.softmax(logits, axis)


class Dropout(object):

    def __init__(self, keep, seed=0):
        self.keep = keep
        self.seed = seed

    def __call__(self, inputs, *args, **kwargs):
        outputs = tf.nn.dropout(inputs, rate=1 - (self.keep), seed=self.seed)
        return outputs


class BiasAdd(object):
    """
    Adds bias to value.

    Parameters
    ----------
    x : tensor
        A Tensor with type float, double, int64, int32, uint8, int16, int8, complex64, or complex128.
    bias : tensor
        Must be the same type as value unless value is a quantized type,
        in which case a different quantized type may be used.
    Returns
    -------
        A Tensor with the same type as value.
    """

    def __init__(self, data_format=None):
        self.data_format = data_format

    def __call__(self, x, bias):
        return tf.nn.bias_add(x, bias, data_format=self.data_format)


def bias_add(x, bias, data_format=None, name=None):
    """
    Adds bias to value.

    Parameters
    ----------
    x : tensor
        A Tensor with type float, double, int64, int32, uint8, int16, int8, complex64, or complex128.
    bias : tensor
        Must be the same type as value unless value is a quantized type,
        in which case a different quantized type may be used.
    data_format : A string.
        'N...C' and 'NC...' are supported.
    name : str
        A name for the operation (optional).
    Returns
    -------
        A Tensor with the same type as value.
    """

    x = tf.nn.bias_add(x, bias, data_format=data_format, name=name)
    return x


class Conv1D(object):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        self.stride = stride
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        outputs = tf.nn.conv1d(
            input=input,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
            # name=name
        )
        return outputs


def conv1d(input, filters, stride, padding, data_format='NWC', dilations=None):
    """
    Computes a 1-D convolution given 3-D input and filter tensors.

    Parameters
    ----------
    input : tensor
        A 3D Tensor. Must be of type float16, float32, or float64
    filters : tensor
        A 3D Tensor. Must have the same type as input.
    stride : int of list
         An int or list of ints that has length 1 or 3. The number of entries by which the filter is moved right at each step.
    padding : string
         'SAME' or 'VALID'
    data_format : string
        An optional string from "NWC", "NCW". Defaults to "NWC", the data is stored in the order of
        [batch, in_width, in_channels]. The "NCW" format stores data as [batch, in_channels, in_width].
    dilations : int or list
        An int or list of ints that has length 1 or 3 which defaults to 1.
        The dilation factor for each dimension of input. If set to k > 1,
        there will be k-1 skipped cells between each filter element on that dimension.
        Dilations in the batch and depth dimensions must be 1.
    name : string
        A name for the operation (optional).
    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_1d_format(data_format, padding)
    outputs = tf.nn.conv1d(
        input=input,
        filters=filters,
        stride=stride,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        # name=name
    )
    return outputs


class Conv2D(object):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, input, filters):
        outputs = tf.nn.conv2d(
            input=input,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None):
    """
    Computes a 2-D convolution given 4-D input and filters tensors.

    Parameters
    ----------
    input : tensor
        Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor.
        The dimension order is interpreted according to the value of data_format, see below for details.
    filters : tensor
         Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    strides : int of list
        The stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension.
        By default the N and C dimensions are set to 1. The dimension order is determined by the value of data_format, see below for details.
    padding : string
        "SAME" or "VALID"
    data_format : string
        "NHWC", "NCHW". Defaults to "NHWC".
    dilations : list or ints
        list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension ofinput.
    name : string
         A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.conv2d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )
    return outputs


class Conv3D(object):

    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        outputs = tf.nn.conv3d(
            input=input,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None):
    """
    Computes a 3-D convolution given 5-D input and filters tensors.

    Parameters
    ----------
    input : tensor
        Must be one of the following types: half, bfloat16, float32, float64.
        Shape [batch, in_depth, in_height, in_width, in_channels].
    filters : tensor
        Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels].
        in_channels must match between input and filters.
    strides : list of ints
        A list of ints that has length >= 5. 1-D tensor of length 5.
        The stride of the sliding window for each dimension of input.
        Must have strides[0] = strides[4] = 1.
    padding : string
        A string from: "SAME", "VALID". The type of padding algorithm to use.
    data_format : string
        An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
        With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
        Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
    dilations : list of ints
        Defaults to [1, 1, 1, 1, 1]. 1-D tensor of length 5. The dilation factor for each dimension of input.
        If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension.
        The dimension order is determined by the value of data_format, see above for details.
        Dilations in the batch and depth dimensions must be 1.
    name : string
        A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d(
        input=input,
        filters=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,  # 'NDHWC',
        dilations=dilations,  # [1, 1, 1, 1, 1],
        # name=name,
    )
    return outputs


def lrn(inputs, depth_radius, bias, alpha, beta):
    """
    Local Response Normalization.

    Parameters
    ----------
    inputs : tensor
        Must be one of the following types: half, bfloat16, float32. 4-D.
    depth_radius : int
        Defaults to 5. 0-D. Half-width of the 1-D normalization window.
    bias : float
        Defaults to 1. An offset (usually positive to avoid dividing by 0).
    alpha : float
        Defaults to 1. A scale factor, usually positive.
    beta : float
         Defaults to 0.5. An exponent.

    Returns
    -------
        A Tensor. Has the same type as input.
    """

    outputs = tf.nn.lrn(inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
    return outputs


def moments(x, axes, shift=None, keepdims=False):
    """
    Calculates the mean and variance of x.

    Parameters
    ----------
    x : tensor
        A Tensor
    axes : list or ints
        Axes along which to compute mean and variance.
    shift : int
        Not used in the current implementation.
    keepdims : bool
        produce moments with the same dimensionality as the input.

    Returns
    -------
        Two Tensor objects: mean and variance.
    """

    outputs = tf.nn.moments(x, axes, shift, keepdims)
    return outputs


class MaxPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = tf.nn.max_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


class MaxPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.data_format = data_format
        self.padding = padding

    def __call__(self, inputs):
        if inputs.ndim == 3:
            self.data_format, self.padding = preprocess_1d_format(data_format=self.data_format, padding=self.padding)
        elif inputs.ndim == 4:
            self.data_format, self.padding = preprocess_2d_format(data_format=self.data_format, padding=self.padding)
        elif inputs.ndim == 5:
            self.data_format, self.padding = preprocess_3d_format(data_format=self.data_format, padding=self.padding)

        outputs = tf.nn.max_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


def max_pool(input, ksize, strides, padding, data_format=None):
    """
    Performs the max pooling on the input.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels] if data_format does not start
        with "NC" (default), or [batch_size, num_channels] + input_spatial_shape if data_format starts with "NC".
        Pooling happens over the spatial dimensions only.
    ksize : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    name : string
        A name for the operation (optional).

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """

    if input.ndim == 3:
        data_format, padding = preprocess_1d_format(data_format=data_format, padding=padding)
    elif input.ndim == 4:
        data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)
    elif input.ndim == 5:
        data_format, padding = preprocess_3d_format(data_format=data_format, padding=padding)

    outputs = tf.nn.max_pool(input=input, ksize=ksize, strides=strides, padding=padding, data_format=data_format)
    return outputs


class AvgPool1d(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = tf.nn.pool(
            input=inputs,
            window_shape=self.ksize,
            pooling_type="AVG",
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


class AvgPool(object):

    def __init__(self, ksize, strides, padding, data_format=None):
        self.ksize = ksize
        self.strides = strides
        self.data_format = data_format
        self.padding = padding_format(padding)

    def __call__(self, inputs):
        outputs = tf.nn.avg_pool(
            input=inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, data_format=self.data_format
        )
        return outputs


def avg_pool(input, ksize, strides, padding):
    """
    Performs the avg pooling on the input.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
        if data_format does not start with "NC" (default), or [batch_size, num_channels] + input_spatial_shape
        if data_format starts with "NC". Pooling happens over the spatial dimensions only.
    ksize : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    name : string
        Optional name for the operation.

    Returns
    -------
        A Tensor of format specified by data_format. The average pooled output tensor.
    """

    padding = padding_format(padding)
    outputs = tf.nn.avg_pool(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
    )
    return outputs


class MaxPool3d(object):
    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = tf.nn.max_pool3d(
            input=inputs,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


def max_pool3d(input, ksize, strides, padding, data_format=None):
    """
    Performs the max pooling on the input.

    Parameters
    ----------
    input : tensor
         A 5-D Tensor of the format specified by data_format.
    ksize : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
         "NDHWC", "NCDHW". Defaults to "NDHWC". The data format of the input and output data.
         With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels].
         Alternatively, the format could be "NCDHW", the data storage order is: [batch, in_channels, in_depth, in_height, in_width].
    name : string
         A name for the operation (optional).

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.max_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    return outputs


class AvgPool3d(object):
    def __init__(self, ksize, strides, padding, data_format=None):
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        self.ksize = ksize
        self.strides = strides

    def __call__(self, inputs):
        outputs = tf.nn.avg_pool3d(
            input=inputs,
            ksize=self.ksize,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


def avg_pool3d(input, ksize, strides, padding, data_format=None):
    """
    Performs the average pooling on the input.

    Parameters
    ----------
    input : tensor
        A 5-D Tensor of shape [batch, height, width, channels] and type float32, float64, qint8, quint8, or qint32.
    ksize : int or list of ints
        An int or list of ints that has length 1, 3 or 5. The size of the window for each dimension of the input tensor.
    strides : int or list of ints
        An int or list of ints that has length 1, 3 or 5.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NDHWC' and 'NCDHW' are supported.
    name : string
        Optional name for the operation.

    Returns
    -------
        A Tensor with the same type as value. The average pooled output tensor.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.avg_pool3d(
        input=input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    return outputs


def pool(input, window_shape, pooling_type, strides=None, padding='VALID', data_format=None, dilations=None, name=None):
    """
    Performs an N-D pooling operation.

    Parameters
    ----------
    input : tensor
        Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
        if data_format does not start with "NC" (default), or [batch_size, num_channels] + input_spatial_shape
        if data_format starts with "NC". Pooling happens over the spatial dimensions only.
    window_shape : int
        Sequence of N ints >= 1.
    pooling_type : string
        Specifies pooling operation, must be "AVG" or "MAX".
    strides : ints
        Sequence of N ints >= 1. Defaults to [1]*N. If any value of strides is > 1, then all values of dilation_rate must be 1.
    padding : string
        The padding algorithm, must be "SAME" or "VALID". Defaults to "SAME".
        See the "returns" section of tf.ops.convolution for details.
    data_format : string
        Specifies whether the channel dimension of the input and output is the last dimension (default, or if data_format does not start with "NC"),
        or the second dimension (if data_format starts with "NC").
        For N=1, the valid values are "NWC" (default) and "NCW". For N=2, the valid values are "NHWC" (default) and "NCHW".
        For N=3, the valid values are "NDHWC" (default) and "NCDHW".
    dilations : list of ints
        Dilation rate. List of N ints >= 1. Defaults to [1]*N. If any value of dilation_rate is > 1, then all values of strides must be 1.
    name : string
        Optional. Name of the op.

    Returns
    -------
        Tensor of rank N+2, of shape [batch_size] + output_spatial_shape + [num_channels]
    """
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


class DepthwiseConv2d(object):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    def __call__(self, input, filter):
        outputs = tf.nn.depthwise_conv2d(
            input=input,
            filter=filter,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def depthwise_conv2d(input, filter, strides, padding, data_format=None, dilations=None, name=None):
    """
    Depthwise 2-D convolution.

    Parameters
    ----------
    input : tensor
        4-D with shape according to data_format.
    filter : tensor
        4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    strides : list
        1-D of size 4. The stride of the sliding window for each dimension of input.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        The data format for input. Either "NHWC" (default) or "NCHW".
    dilations : list
        1-D of size 2. The dilation rate in which we sample input values across the height and width dimensions in atrous convolution.
        If it is greater than 1, then all values of strides must be 1.
    name : string
        A name for this operation (optional).

    Returns
    -------
        A 4-D Tensor with shape according to data_format.
        E.g., for "NHWC" format, shape is [batch, out_height, out_width, in_channels * channel_multiplier].
    """

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


class Conv1d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None, in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

    def __call__(self, input, filters):
        batch_size = input.shape[0]
        if self.data_format == 'NWC':
            w_axis, c_axis = 1, 2
        else:
            w_axis, c_axis = 2, 1

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        input_w = input_shape[w_axis]
        filters_w = filters_shape[0]
        output_channels = filters_shape[1]
        dilations_w = 1

        if isinstance(self.strides, int):
            strides_w = self.strides
        else:
            strides_list = list(self.strides)
            strides_w = strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                dilations_w = dilations_list[w_axis]

        filters_w = filters_w + (filters_w - 1) * (dilations_w - 1)
        assert self.padding in {'SAME', 'VALID'}
        if self.padding == 'VALID':
            output_w = input_w * strides_w + max(filters_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_w = input_w * strides_w

        if self.data_format == 'NCW':
            output_shape = (batch_size, output_channels, output_w)
        else:
            output_shape = (batch_size, output_w, output_channels)
        output_shape = tf.stack(output_shape)
        outputs = tf.nn.conv1d_transpose(
            input=input,
            filters=filters,
            output_shape=output_shape,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )
        return outputs


def conv1d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NWC', dilations=None, name=None
):
    """
    The transpose of conv1d.

    Parameters
    ----------
    input : tensor
        A 3-D Tensor of type float and shape [batch, in_width, in_channels]
        for NWC data format or [batch, in_channels, in_width] for NCW data format.
    filters : tensor
        A 3-D Tensor with the same type as value and shape [filter_width, output_channels, in_channels].
        filter's in_channels dimension must match that of value.
    output_shape : tensor
        A 1-D Tensor, containing three elements, representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1 or 3. The number of entries by which the filter is moved right at each step.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NWC' and 'NCW' are supported.
    dilations : list
         An int or list of ints that has length 1 or 3 which defaults to 1.
         The dilation factor for each dimension of input. If set to k > 1,
         there will be k-1 skipped cells between each filter element on that dimension.
         Dilations in the batch and depth dimensions must be 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as value.
    """

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


class Conv2d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NHWC':
            h_axis, w_axis = 1, 2
        else:
            h_axis, w_axis = 2, 3

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        batch_size = input.shape[0]
        input_h, input_w = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = filters_shape[0], filters_shape[1]
        output_channels = filters_shape[2]
        dilations_h, dilations_w = 1, 1

        if isinstance(self.strides, int):
            strides_h = self.strides
            strides_w = self.strides
        else:
            strides_list = list(self.strides)
            if len(strides_list) != 4:
                strides_h = strides_list[0]
                strides_w = strides_list[1]
            else:
                strides_h = strides_list[h_axis]
                strides_w = strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_h = self.dilations
                dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                if len(dilations_list) != 4:
                    dilations_h = dilations_list[0]
                    dilations_w = dilations_list[1]
                else:
                    dilations_h = dilations_list[h_axis]
                    dilations_w = dilations_list[w_axis]

        kernel_h = kernel_h + (kernel_h - 1) * (dilations_h - 1)
        kernel_w = kernel_w + (kernel_w - 1) * (dilations_w - 1)

        assert self.padding in {'SAME', 'VALID'}
        if self.padding == 'VALID':
            output_h = input_h * strides_h + max(kernel_h - strides_h, 0)
            output_w = input_w * strides_w + max(kernel_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_h = input_h * strides_h
            output_w = input_w * strides_w

        if self.data_format == 'NCHW':
            out_shape = (batch_size, output_channels, output_h, output_w)
        else:
            out_shape = (batch_size, output_h, output_w, output_channels)

        output_shape = tf.stack(out_shape)

        outputs = tf.nn.conv2d_transpose(
            input=input, filters=filters, output_shape=output_shape, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=self.dilations, name=self.name
        )
        return outputs


def conv2d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NHWC', dilations=None, name=None
):
    """
    The transpose of conv2d.

    Parameters
    ----------
    input : tensor
        A 4-D Tensor of type float and shape [batch, height, width, in_channels]
        for NHWC data format or [batch, in_channels, height, width] for NCHW data format.
    filters : tensor
        A 4-D Tensor with the same type as input and shape [height, width,
        output_channels, in_channels]. filter's in_channels dimension must match that of input.
    output_shape : tensor
        A 1-D Tensor representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of input.
        If a single value is given it is replicated in the H and W dimension.
        By default the N and C dimensions are set to 0.
        The dimension order is determined by the value of data_format, see below for details.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
         'NHWC' and 'NCHW' are supported.
    dilations : list
        An int or list of ints that has length 1, 2 or 4, defaults to 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as input.
    """

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
    return outputs


class Conv3d_transpose(object):

    def __init__(
        self, strides, padding, data_format='NDHWC', dilations=None, name=None, out_channel=None, k_size=None,
        in_channels=None
    ):
        self.strides = strides
        self.dilations = dilations
        self.name = name
        self.out_channel = out_channel
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

    def __call__(self, input, filters):
        if self.data_format == 'NDHWC':
            d_axis, h_axis, w_axis = 1, 2, 3
        else:
            d_axis, h_axis, w_axis = 2, 3, 4

        input_shape = input.shape.as_list()
        filters_shape = filters.shape.as_list()
        batch_size = input_shape[0]
        input_d, input_h, input_w = input_shape[d_axis], input_shape[h_axis], input_shape[w_axis]
        kernel_d, kernel_h, kernel_w = filters_shape[0], filters_shape[1], filters_shape[2]
        dilations_d, dilations_h, dilations_w = 1, 1, 1

        if isinstance(self.strides, int):
            strides_d, strides_h, strides_w = self.strides
        else:
            strides_list = list(self.strides)
            if len(strides_list) != 5:
                strides_d, strides_h, strides_w = \
                    strides_list[0], \
                    strides_list[1], \
                    strides_list[2]
            else:
                strides_d, strides_h, strides_w = \
                    strides_list[d_axis], \
                    strides_list[h_axis], \
                    strides_list[w_axis]

        if self.dilations is not None:
            if isinstance(self.dilations, int):
                dilations_d, dilations_h, dilations_w = self.dilations
            else:
                dilations_list = list(self.dilations)
                if len(dilations_list) != 5:
                    dilations_d, dilations_h, dilations_w = \
                        dilations_list[0], \
                        dilations_list[1], \
                        dilations_list[2]
                else:
                    dilations_d, dilations_h, dilations_w = \
                        dilations_list[d_axis],\
                        dilations_list[h_axis], \
                        dilations_list[w_axis]

        assert self.padding in {'VALID', 'SAME'}

        kernel_d = kernel_d + (kernel_d - 1) * (dilations_d - 1)
        kernel_h = kernel_h + (kernel_h - 1) * (dilations_h - 1)
        kernel_w = kernel_w + (kernel_w - 1) * (dilations_w - 1)

        if self.padding == 'VALID':
            output_d = input_d * strides_d + max(kernel_d - strides_d, 0)
            output_h = input_h * strides_h + max(kernel_h - strides_h, 0)
            output_w = input_w * strides_w + max(kernel_w - strides_w, 0)
        elif self.padding == 'SAME':
            output_d = input_d * strides_d
            output_h = input_h * strides_h
            output_w = input_w * strides_w

        if self.data_format == 'NDHWC':
            output_shape = (batch_size, output_d, output_h, output_w, self.out_channel)
        else:
            output_shape = (batch_size, self.out_channel, output_d, output_h, output_w)

        output_shape = tf.stack(output_shape)
        outputs = tf.nn.conv3d_transpose(
            input=input, filters=filters, output_shape=output_shape, strides=self.strides, padding=self.padding,
            data_format=self.data_format, dilations=self.dilations, name=self.name
        )

        return outputs


def conv3d_transpose(
    input, filters, output_shape, strides, padding='SAME', data_format='NDHWC', dilations=None, name=None
):
    """
    The transpose of conv3d.

    Parameters
    ----------
    input : tensor
         A 5-D Tensor of type float and shape [batch, height, width, in_channels] for
         NHWC data format or [batch, in_channels, height, width] for NCHW data format.
    filters : tensor
        A 5-D Tensor with the same type as value and shape [height, width, output_channels, in_channels].
        filter's in_channels dimension must match that of value.
    output_shape : tensor
        A 1-D Tensor representing the output shape of the deconvolution op.
    strides : list
        An int or list of ints that has length 1, 3 or 5.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.
    data_format : string
        'NDHWC' and 'NCDHW' are supported.
    dilations : list of ints
        An int or list of ints that has length 1, 3 or 5, defaults to 1.
    name : string
        Optional name for the returned tensor.

    Returns
    -------
        A Tensor with the same type as value.
    """

    data_format, padding = preprocess_3d_format(data_format, padding)
    outputs = tf.nn.conv3d_transpose(
        input=input, filters=filters, output_shape=output_shape, strides=strides, padding=padding,
        data_format=data_format, dilations=dilations, name=name
    )
    return outputs


def depthwise_conv2d(input, filters, strides, padding='SAME', data_format='NHWC', dilations=None, name=None):
    """
    Depthwise 2-D convolution.

    Parameters
    ----------
    input : tensor
        4-D with shape according to data_format.
    filters : tensor
        4-D with shape [filter_height, filter_width, in_channels, channel_multiplier].
    strides : tuple
        1-D of size 4. The stride of the sliding window for each dimension of input.
    padding : string
        'VALID' or 'SAME'
    data_format : string
        "NHWC" (default) or "NCHW".
    dilations : tuple
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution.
        If it is greater than 1, then all values of strides must be 1.
    name : string
        A name for this operation (optional).

    Returns
    -------
        A 4-D Tensor with shape according to data_format.
    """

    data_format, padding = preprocess_2d_format(data_format, padding)
    outputs = tf.nn.depthwise_conv2d(
        input=input,
        filter=filters,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )
    return outputs


def _to_channel_first_bias(b):
    """Reshape [c] to [c, 1, 1]."""
    channel_size = int(b.shape[0])
    new_shape = (channel_size, 1, 1)
    return tf.reshape(b, new_shape)


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    if data_format == 'NHWC':
        return x * b
    elif data_format == 'NCHW':
        return x * _to_channel_first_bias(b)
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    if data_format == 'NHWC':
        return tf.add(x, b)
    elif data_format == 'NCHW':
        return tf.add(x, _to_channel_first_bias(b))
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    if data_format == 'channels_last':
        mean = tf.reshape(mean, [1] * (len(x.shape) - 1) + [-1])
        variance = tf.reshape(variance, [1] * (len(x.shape) - 1) + [-1])
        offset = tf.reshape(offset, [1] * (len(x.shape) - 1) + [-1])
        scale = tf.reshape(scale, [1] * (len(x.shape) - 1) + [-1])
    elif data_format == 'channels_first':
        mean = tf.reshape(mean, [1] + [-1] + [1] * (len(x.shape) - 2))
        variance = tf.reshape(variance, [1] + [-1] + [1] * (len(x.shape) - 2))
        offset = tf.reshape(offset, [1] + [-1] + [1] * (len(x.shape) - 2))
        scale = tf.reshape(scale, [1] + [-1] + [1] * (len(x.shape) - 2))
    else:
        raise ValueError('invalid data_format: %s' % data_format)

    with ops.name_scope(name, 'batchnorm', [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale

        a = math_ops.cast(inv, x.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype)
        # Return a * x + b with customized data_format.
        # Currently TF doesn't have bias_scale, and tensorRT has bug in converting tf.nn.bias_add
        # So we reimplemted them to allow make the model work with tensorRT.
        # See https://github.com/tensorlayer/openpose-plus/issues/75 for more details.
        # df = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
        # return _bias_add(_bias_scale(x, a, df[data_format]), b, df[data_format])
        return a * x + b


class BatchNorm(object):
    """
    The :class:`BatchNorm` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    act : activation function
        The activation function of this layer.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    moving_mean_init : initializer or None
        The initializer for initializing moving mean, if None, skip moving mean.
    moving_var_init : initializer or None
        The initializer for initializing moving var, if None, skip moving var.
    num_features: int
        Number of features for input tensor. Useful to build layer if using BatchNorm1d, BatchNorm2d or BatchNorm3d,
        but should be left as None if using BatchNorm. Default None.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm()(net)

    Notes
    -----
    The :class:`BatchNorm` is universally suitable for 3D/4D/5D input in static model, but should not be used
    in dynamic model where layer is built upon class initialization. So the argument 'num_features' should only be used
    for subclasses :class:`BatchNorm1d`, :class:`BatchNorm2d` and :class:`BatchNorm3d`. All the three subclasses are
    suitable under all kinds of conditions.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    def __init__(
        self, decay=0.9, epsilon=0.00001, beta=None, gamma=None, moving_mean=None, moving_var=None, num_features=None,
        data_format='channels_last', is_train=False
    ):
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta = beta
        self.gamma = gamma
        self.moving_mean = moving_mean
        self.moving_var = moving_var
        self.num_features = num_features
        self.is_train = is_train
        self.axes = None

        if self.decay < 0.0 or 1.0 < self.decay:
            raise ValueError("decay should be between 0 to 1")

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        if inputs.ndim <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(inputs.ndim))

    def __call__(self, inputs):
        self._check_input_shape(inputs)
        self.channel_axis = len(inputs.shape) - 1 if self.data_format == 'channels_last' else 1
        if self.axes is None:
            self.axes = [i for i in range(len(inputs.shape)) if i != self.channel_axis]

        mean, var = tf.nn.moments(inputs, self.axes, keepdims=False)
        if self.is_train:
            # update moving_mean and moving_var
            self.moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay, zero_debias=False
            )
            self.moving_var = moving_averages.assign_moving_average(self.moving_var, var, self.decay, zero_debias=False)
            outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon, self.data_format)
        else:
            outputs = batch_normalization(
                inputs, self.moving_mean, self.moving_var, self.beta, self.gamma, self.epsilon, self.data_format
            )

        return outputs


class GroupConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations
        self.groups = groups
        if self.data_format == 'NHWC':
            self.channels_axis = 3
        else:
            self.channels_axis = 1

    def __call__(self, input, filters):

        if self.groups == 1:
            outputs = tf.nn.conv2d(
                input=input,
                filters=filters,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilations=self.dilations,
            )
        else:
            inputgroups = tf.split(input, num_or_size_splits=self.groups, axis=self.channels_axis)
            weightsgroups = tf.split(filters, num_or_size_splits=self.groups, axis=self.channels_axis)
            convgroups = []
            for i, k in zip(inputgroups, weightsgroups):
                convgroups.append(
                    tf.nn.conv2d(
                        input=i,
                        filters=k,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilations=self.dilations,
                    )
                )
            outputs = tf.concat(axis=self.channels_axis, values=convgroups)

        return outputs


class SeparableConv1D(object):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)

        if self.data_format == 'NWC':
            self.spatial_start_dim = 1
            self.strides = (1, stride, stride, 1)
            self.data_format = 'NHWC'
        else:
            self.spatial_start_dim = 2
            self.strides = (1, 1, stride, stride)
            self.data_format = 'NCHW'
        self.dilation_rate = (1, dilations)

    def __call__(self, inputs, depthwise_filters, pointwise_filters):
        inputs = tf.expand_dims(inputs, axis=self.spatial_start_dim)
        depthwise_filters = tf.expand_dims(depthwise_filters, 0)
        pointwise_filters = tf.expand_dims(pointwise_filters, 0)

        outputs = tf.nn.separable_conv2d(
            inputs, depthwise_filters, pointwise_filters, strides=self.strides, padding=self.padding,
            dilations=self.dilation_rate, data_format=self.data_format
        )

        outputs = tf.squeeze(outputs, axis=self.spatial_start_dim)

        return outputs


class SeparableConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = (dilations[2], dilations[2])

    def __call__(self, inputs, depthwise_filters, pointwise_filters):

        outputs = tf.nn.separable_conv2d(
            inputs, depthwise_filters, pointwise_filters, strides=self.strides, padding=self.padding,
            dilations=self.dilations, data_format=self.data_format
        )

        return outputs


class AdaptiveMeanPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NWC':
            n, w, c = input.shape
        else:
            n, c, w = input.shape

        stride = floor(w / self.output_size)
        kernel = w - (self.output_size - 1) * stride
        output = tf.nn.avg_pool1d(input, ksize=kernel, strides=stride, data_format=self.data_format, padding='VALID')

        return output


class AdaptiveMeanPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.avg_pool2d(
            inputs, ksize=(kernel_h, kernel_w), strides=(stride_h, stride_w), data_format=self.data_format,
            padding='VALID'
        )

        return outputs


class AdaptiveMeanPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NDHWC':
            n, d, h, w, c = inputs.shape
        else:
            n, c, d, h, w = inputs.shape

        out_d, out_h, out_w = self.output_size
        stride_d = floor(d / out_d)
        kernel_d = d - (out_d - 1) * stride_d
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.avg_pool3d(
            inputs, ksize=(kernel_d, kernel_h, kernel_w), strides=(stride_d, stride_h, stride_w),
            data_format=self.data_format, padding='VALID'
        )

        return outputs


class AdaptiveMaxPool1D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, input):

        if self.data_format == 'NWC':
            n, w, c = input.shape
        else:
            n, c, w = input.shape

        stride = floor(w / self.output_size)
        kernel = w - (self.output_size - 1) * stride
        output = tf.nn.max_pool1d(input, ksize=kernel, strides=stride, data_format=self.data_format, padding='VALID')

        return output


class AdaptiveMaxPool2D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.max_pool2d(
            inputs, ksize=(kernel_h, kernel_w), strides=(stride_h, stride_w), data_format=self.data_format,
            padding='VALID'
        )

        return outputs


class AdaptiveMaxPool3D(object):

    def __init__(self, output_size, data_format):
        self.data_format, _ = preprocess_3d_format(data_format, None)
        self.output_size = output_size

    def __call__(self, inputs):

        if self.data_format == 'NDHWC':
            n, d, h, w, c = inputs.shape
        else:
            n, c, d, h, w = inputs.shape

        out_d, out_h, out_w = self.output_size
        stride_d = floor(d / out_d)
        kernel_d = d - (out_d - 1) * stride_d
        stride_h = floor(h / out_h)
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = floor(w / out_w)
        kernel_w = w - (out_w - 1) * stride_w

        outputs = tf.nn.max_pool3d(
            inputs, ksize=(kernel_d, kernel_h, kernel_w), strides=(stride_d, stride_h, stride_w),
            data_format=self.data_format, padding='VALID'
        )

        return outputs


class BinaryConv2D(object):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations

    # @tf.RegisterGradient("TL_Sign_QuantizeGrad")
    # def _quantize_grad(op, grad):
    #     """Clip and binarize tensor using the straight through estimator (STE) for the gradient."""
    #     return tf.clip_by_value(grad, -1, 1)

    def quantize(self, x):
        # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
        #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
        with tf.compat.v1.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
            return tf.sign(x)

    def __call__(self, inputs, filters):

        filters = self.quantize(filters)

        outputs = tf.nn.conv2d(
            input=inputs, filters=filters, strides=self.strides, padding=self.padding, data_format=self.data_format,
            dilations=self.dilations
        )

        return outputs


class DorefaConv2D(object):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.strides = strides
        self.dilations = dilations
        self.bitW = bitW
        self.bitA = bitA

    def _quantize_dorefa(self, x, k):
        G = tf.compat.v1.get_default_graph()
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def cabs(self, x):
        return tf.minimum(1.0, tf.abs(x), name='cabs')

    def quantize_active(self, x, bitA):
        if bitA == 32:
            return x
        return self._quantize_dorefa(x, bitA)

    def quantize_weight(self, x, bitW, force_quantization=False):

        G = tf.compat.v1.get_default_graph()
        if bitW == 32 and not force_quantization:
            return x
        if bitW == 1:  # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(input_tensor=tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.clip_by_value(
            x * 0.5 + 0.5, 0.0, 1.0
        )  # it seems as though most weights are within -1 to 1 region anyways
        return 2 * self._quantize_dorefa(x, bitW) - 1

    def __call__(self, inputs, filters):

        inputs = self.quantize_active(self.cabs(inputs), self.bitA)

        filters = self.quantize_weight(filters, self.bitW)

        outputs = tf.nn.conv2d(
            input=inputs,
            filters=filters,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilations,
        )

        return outputs
