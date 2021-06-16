#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import mindspore as ms
import mindspore.ops as P
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore._checkparam import Rel
from mindspore.ops import functional as F
from mindspore.communication import management
from mindspore.ops.operations import _inner_ops as inner
from mindspore._extends import cell_attr_register
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore._checkparam import Validator as validator
from mindspore.communication.management import get_group_size, get_rank

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
        padding = "same"
    elif padding in ["VALID", "valid"]:
        padding = "valid"
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

    if data_format in ["channels_last", "NHWC", "nhwc"]:
        data_format = "NHWC"
    elif data_format in ["channels_first", "NCHW", "nchw"]:
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

    if len(P.Shape()(x)) == 3:
        x = P.Transpose()(x, (0, 2, 1))
    elif len(P.Shape()(x)) == 4:
        x = P.Transpose()(x, (0, 2, 3, 1))
    elif len(P.Shape()(x)) == 5:
        x = P.Transpose()(x, (0, 2, 3, 4, 1))
    # else:
    #     raise Exception("Unsupported dimensions")
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

    if len(P.Shape()(x)) == 3:
        x = P.Transpose()(x, (0, 2, 1))
    elif len(P.Shape()(x)) == 4:
        x = P.Transpose()(x, (0, 3, 1, 2))
    elif len(P.Shape()(x)) == 5:
        x = P.Transpose()(x, (0, 4, 1, 2, 3))
    # else:
    #     raise Exception("Unsupported dimensions")
    return x


class ReLU(Cell):

    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


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
    outputs = P.ReLU()
    return outputs(x)


class ReLU6(Cell):

    def __init__(self):
        super(ReLU6, self).__init__()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        return self.relu6(x)


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
    outputs = P.ReLU6()
    return outputs(x)


class LeakyReLU(Cell):

    def __init__(self, alpha=0.2):
        super(LeakyReLU, self).__init__()
        self.leakyrelu = ms.nn.LeakyReLU(alpha=alpha)

    def construct(self, x):
        return self.leakyrelu(x)


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

    leaky_relu = LeakyReLU(alpha=alpha)
    output = leaky_relu(x)
    return leaky_relu


class Softplus(Cell):

    def __init__(self):
        super(Softplus, self).__init__()
        self.softplus = P.Softplus()

    def construct(self, x):
        return self.softplus(x)


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

    obj = Softplus()
    return obj(x)


class Tanh(Cell):

    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = P.Tanh()

    def construct(self, x):
        return self.tanh(x)


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

    _tanh = Tanh()
    return _tanh(x)


class Sigmoid(Cell):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        return self.sigmoid(x)


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
    outputs = P.Sigmoid()
    return outputs(x)


class Softmax(Cell):

    def __init__(self):
        super(Softmax, self).__init__()
        self.softmax = P.Softmax()

    def construct(self, x):
        return self.softmax(x)


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
    outputs = P.Softmax(axis)
    return outputs(logits)


class Dropout(Cell):

    def __init__(self, keep, seed=0):
        super(Dropout, self).__init__()
        self.dropout = P.Dropout(keep_prob=keep)
        self.is_gpu = context.get_context('device_target') in ["GPU"]
        self.get_shape = P.Shape()
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=seed, Seed1=0)
        self.dropout_do_mask = P.DropoutDoMask()
        self.cast = P.Cast()
        self.keep_prob = keep  # ms.Tensor(keep, dtype=ms.float32)
        # print(self.keep_prob, type(self.keep_prob))

    def construct(self, inputs):
        if self.is_gpu:
            outputs, _ = self.dropout(inputs)
            return outputs
        if self.keep_prob == 1:
            return inputs
        shape = self.get_shape(inputs)
        dtype = P.DType()(inputs)
        if self._is_float_dtype(dtype):
            keep_prob = self.cast(self.keep_prob, dtype=dtype)
        else:
            keep_prob = self.cast(self.keep_prob, ms.float16)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(inputs, output, keep_prob)

    def _is_float_dtype(dtype):
        if dtype in [ms.float32, ms.float16]:
            return True
        return False


class BiasAdd(Cell):
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

    def __init__(self, data_format='channels_first'):
        super(BiasAdd, self).__init__()
        self.bias_add = P.BiasAdd()
        if data_format in ['channels_first', 'NCW', 'NCHW', 'NCDHW']:
            self.data_format = 'channels_first'
        elif data_format in ['channels_last', 'NWC', 'NHWC', 'NDHWC']:
            self.data_format = 'channels_last'
        else:
            raise ("Unsupported data format: " + str(data_format))

    def construct(self, x, bias):
        if self.data_format == 'channels_last':
            x = nhwc_to_nchw(x)
        outputs = self.bias_add(x, bias)
        if self.data_format == 'channels_last':
            outputs = nchw_to_nhwc(outputs)
        return outputs


def bias_add(x, bias):
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
    raise NotImplementedError


class Conv1D(Cell):

    def __init__(self, stride, padding, data_format='NWC', dilations=None, out_channel=None, k_size=None):
        super(Conv1D, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.stride = (1, stride)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)
        self.out_channel = out_channel

        self.conv2d = P.Conv2D(
            out_channel=self.out_channel, kernel_size=self.k_size, pad_mode=self.padding, stride=self.stride,
            dilation=self.dilations, mode=1, group=1
        )

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x, filters):
        if self.data_format == 'NWC':
            x = nhwc_to_nchw(x)

        x = self.expand_dims(x, 2)
        filters = self.expand_dims(filters, 2)

        output = self.conv2d(x, filters)
        output = self.squeeze(output)

        if self.data_format == 'NWC':
            output = nchw_to_nhwc(output)
        return output


def conv1d(input, filters, stride, padding, data_format='NWC', dilations=None, name=None):
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

    pass


class Conv2D(Cell):

    def __init__(self, strides, padding, data_format='NHWC', dilations=None, out_channel=None, k_size=None):
        super(Conv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1, data_format=self.data_format
        )

    def construct(self, inputs, filters):
        outputs = self.conv2d(inputs, filters)
        return outputs


def conv2d(input, filters, strides, padding, data_format='NCHW', dilations=None):
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
        "NHWC", "NCHW". Defaults to "NCHW".
    dilations : list or ints
        list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension ofinput.

    Returns
    -------
        A Tensor. Has the same type as input.
    """
    raise NotImplementedError


class Conv3D(Cell):
    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, out_channel=None, k_size=None):
        super(Conv3D, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)

        if self.data_format is 'NDHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
            raise NotImplementedError("The optional value for data format. Currently only support “NCDHW”.")
        elif self.data_format is 'NCDHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv3d = P.Conv3D(out_channel=out_channel,
                               kernel_size=k_size,
                               pad_mode=self.padding,
                               stride=self.ms_stride,
                               dilation=self.ms_dilation,
                               data_format=data_format)

    def construct(self, input, filters):
        outputs = self.conv3d(input, filters)
        return outputs


def conv3d(input, filters, strides, padding, data_format='NDHWC', dilations=None, name=None):
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

    raise NotImplementedError


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
    pass


def moments(x, axes, shift=None, keepdims=False):
    """
    Calculates the mean and variance of x.

    Parameters
    ----------
    x : tensor
        A Tensor
    axes : ints
        Axes along which to compute mean and variance.
    shift : int
        Not used in the current implementation.
    keepdims : bool
        produce moments with the same dimensionality as the input.

    Returns
    -------
        Two Tensor objects: mean and variance.
    """

    pass


class MaxPool1d(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(MaxPool1d, self).__init__()
        self.data_format, padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.expand = P.ExpandDims()
        _strides = (1, strides[0])
        _ksize = (1, ksize[0])
        if self.data_format == 'NWC':
            self.squeeze = P.Squeeze(1)
            _data_format = 'NHWC'
        if self.data_format == 'NCW':
            self.squeeze = P.Squeeze(2)
            _data_format = 'NCHW'

        self.max_pool = P.MaxPool(
            kernel_size=_ksize,
            strides=_strides,
            pad_mode=padding,
            data_format=_data_format
        )

    def construct(self, inputs):
        if self.data_format == 'NWC':
            x = self.expand(inputs, 1)
        if self.data_format == 'NCW':
            x = self.expand(inputs, 2)
        output = self.max_pool(x)
        output = self.squeeze(output)
        return output


class MaxPool(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(MaxPool, self).__init__()
        data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)

        if data_format == 'NHWC':
            _strides = (strides[1], strides[2])
        if data_format == 'NCHW':
            _strides = (strides[2], strides[3])

        self.maxpool = P.MaxPool(
            kernel_size = ksize,
            strides = _strides,
            pad_mode = padding,
            data_format = data_format
        )

    def construct(self, inputs):
        outputs = self.maxpool(inputs)
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
    strides : list or list of ints
        An int or list of ints that has length 1, N or N+2.
        The stride of the sliding window for each dimension of the input tensor.
    padding : string
        'VALID' or 'SAME'. The padding algorithm. See the "returns" section of tf.ops.convolution for details.

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """
    data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)
    if data_format == 'NHWC':
        _strides = (strides[1], strides[2])
    if data_format == 'NCHW':
        _strides = (strides[2], strides[3])
    outputs = P.MaxPool(
        kernel_size=ksize,
        strides=_strides,
        pad_mode=padding,
        data_format=data_format
    )(input)
    return outputs



class AvgPool1d(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(AvgPool1d, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format=data_format, padding=padding)
        self.kernel_size = (1, ksize[0])
        self.stride = (1, strides[0])

        if self.data_format == 'NWC':
            _data_format = 'NHWC'
            self.squeeze = P.Squeeze(1)
        if self.data_format == 'NCW':
            _data_format = 'NCHW'
            self.squeeze = P.Squeeze(2)

        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.padding,
                                  data_format=_data_format)
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.slice = P.Slice()
        self.expand = P.ExpandDims()
        self.shape = P.Shape()

    def construct(self, inputs):
        x = inputs
        batch, channel, width = self.shape(inputs)
        if width == self.kernel_size[1]:
            x = self.reduce_mean(x, 2)
        elif width - self.kernel_size[1] < self.stride[1]:
            x = self.slice(x, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            x = self.reduce_mean(x, 2)
        else:
            if self.data_format == 'NCW':
                x = self.expand(x, 2)
            if self.data_format == 'NWC':
                x = self.expand(x, 1)
            x = self.avg_pool(x)
            x = self.squeeze(x)
        return x


class AvgPool(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(AvgPool, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format=data_format, padding=padding)
        ms_ksize = ksize[1]
        ms_strides = strides[1]
        self.avgpool = P.AvgPool(ksize=ms_ksize, strides=ms_strides, padding=padding, data_format=self.data_format)

    def construct(self, inputs):
        outputs = self.avgpool(inputs)
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

    Returns
    -------
        A Tensor of format specified by data_format. The average pooled output tensor.
    """
    padding = padding_format(padding)
    ms_ksize = ksize[0]
    ms_strides = strides[1]
    outputs = P.AvgPool(ksize=ms_ksize, strides=ms_strides, padding=padding)
    return outputs(input)


def max_pool3d(input, ksize, strides, padding, data_format=None, name=None):
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
    pass


def avg_pool3d(input, ksize, strides, padding, data_format=None, name=None):
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
    pass


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
    pass


class DepthwiseConv2d(Cell):

    def __init__(self, strides, padding, data_format=None, dilations=None, ksize=None, channel_multiplier=1):
        super(DepthwiseConv2d, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.ms_stride = strides[1]
        self.ms_dilation = dilations[1]
        self.depthwise_conv2d = P.DepthwiseConv2dNative(
            channel_multiplier=channel_multiplier, kernel_size=ksize, stride=self.ms_stride, dilation=self.ms_dilation
        )

    def construct(self, input, filter):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)
        outputs = self.depthwise_conv2d(input, filter)
        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)
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

    pass


class Conv1d_transpose(Cell):

    def __init__(self, strides, padding, data_format, dilations=None, out_channel=None, k_size=None, in_channels=None):
        super(Conv1d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.strides = (1, strides)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)

        self.conv2d_transpose = P.Conv2DBackpropInput(
            out_channel=self.in_channels, kernel_size=self.k_size, pad_mode=self.padding, stride=self.strides,
            dilation=self.dilations, mode=1, group=1
        )
        self.shape = P.Shape()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size):
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)

        if self.padding == 'same':
            length = input_length * stride_size
        elif self.padding == 'valid':
            length = input_length * stride_size + max(filter_size - stride_size, 0)

        return length

    def construct(self, x, filters):
        if self.data_format == 'NWC':
            x = nhwc_to_nchw(x)
        x = self.expand_dims(x, 2)
        filters = self.expand_dims(filters, 2)
        n, _, h, w = self.shape(x)

        h_out = self._deconv_output_length(h, self.k_size[0], self.strides[0], self.dilations[0])
        w_out = self._deconv_output_length(w, self.k_size[1], self.strides[1], self.dilations[1])
        output = self.conv2d_transpose(x, filters, (n, self.out_channel, h_out, w_out))
        output = self.squeeze(output)

        if self.data_format == 'NWC':
            output = nchw_to_nhwc(output)
        return output


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
    pass


class Conv2d_transpose(Cell):

    def __init__(self, strides, padding, data_format, dilations=None, out_channel=None, k_size=None, in_channels=None):
        super(Conv2d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.in_channels = in_channels
        self.out_channel = out_channel

        self.k_size = k_size
        if self.data_format == 'NHWC':
            self.strides = (strides[1], strides[2])
            self.dilations = (dilations[1], dilations[2])
        elif self.data_format == 'NCHW':
            self.strides = (strides[2], strides[3])
            self.dilations = (dilations[2], dilations[3])

        self.conv2d_transpose = P.Conv2DBackpropInput(
            out_channel=self.in_channels, kernel_size=self.k_size, pad_mode=self.padding, stride=self.strides,
            dilation=self.dilations, mode=1, group=1
        )
        self.shape = P.Shape()

    def _deconv_output_length(self, input_length, filter_size, stride_size, dilation_size):
        length = 0
        filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)

        if self.padding == 'same':
            length = input_length * stride_size
        elif self.padding == 'valid':
            length = input_length * stride_size + max(filter_size - stride_size, 0)

        return length

    def construct(self, x, filters):
        if self.data_format == 'NHWC':
            x = nhwc_to_nchw(x)

        n, _, h, w = self.shape(x)

        h_out = self._deconv_output_length(h, self.k_size[0], self.strides[0], self.dilations[0])
        w_out = self._deconv_output_length(w, self.k_size[1], self.strides[1], self.dilations[1])

        output = self.conv2d_transpose(x, filters, (n, self.out_channel, h_out, w_out))

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)

        return output


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
    pass


class Conv3d_transpose(Cell):
    def __init__(self, strides, padding, data_format='NDHWC', dilations=None, name=None, out_channel=None, k_size=None,
            in_channels=None
    ):
        super(Conv3d_transpose, self).__init__()
        self.data_format, self.padding = preprocess_3d_format(data_format, padding)
        if self.data_format == 'NDHWC':
            self.strides = (strides[1], strides[2], strides[3])
            self.dilations = (dilations[1], dilations[2], dilations[3])
        elif self.data_format == 'NCDHW':
            self.strides = (strides[2], strides[3], strides[4])
            self.dilations = (dilations[2], dilations[3], dilations[4])

        self.conv3d_transpose = P.Conv3DTranspose(
            in_channel=in_channels,
            out_channel=out_channel,
            kernel_size=k_size,
            mode=1,
            pad_mode=padding,
            stride=self.strides,
            dilation=self.dilations,
            data_format=self.data_format)

    def construct(self, input, filters):
        output = self.conv3d_transpose(input, filters)
        return output



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

    pass


class BatchNorm(Cell):
    """Batch Normalization base class."""

    @cell_attr_register
    def __init__(self,
                 num_features,
                 epsilon=1e-5,
                 decay=0.9,
                 gamma=None,
                 beta = None,
                 moving_mean = None,
                 moving_var = None,
                 is_train = None,
                 device_num_each_group=1,
                 process_groups=0,
                 data_format='NCHW'):
        super(BatchNorm, self).__init__()
        if data_format in ["channels_last", "NHWC", "nhwc"]:
            data_format = "NHWC"
        elif data_format in ["channels_first", "NCHW", "nchw"]:
            data_format = "NCHW"
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if decay < 0 or decay > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(decay))
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
        self.use_batch_statistics = is_train
        self.num_features = num_features
        self.eps = epsilon
        self.moving_mean = moving_mean
        self.moving_variance = moving_var
        self.gamma = gamma
        self.beta = beta
        self.group_device_num = validator.check_positive_int(device_num_each_group)
        self.process_groups = process_groups
        self.is_global = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        global SYNC_BN_GROUP_NAME
        # for GlobalBatchNorm
        if self.group_device_num != 1:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            self.device_list = [i for i in range(0, self.rank_size)]
            self.rank_list = self.list_group(self.device_list, self.group_device_num)
            self.rank_list_idx = len(self.rank_list)
            for i in range(self.rank_list_idx):
                if self.rank_id in self.rank_list[i]:
                    self.is_global = True
                    if SYNC_BN_GROUP_NAME == "":
                        SYNC_BN_GROUP_NAME = "sync_bn_group"+ str(i)
                        management.create_group(SYNC_BN_GROUP_NAME, self.rank_list[i])
        # for SyncBatchNorm
        if self.process_groups != 0:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            if self.process_groups is not None:
                validator.check_isinstance("process_groups", self.process_groups, list)
                self._check_rank_ids(self.process_groups, self.rank_size)
                for i in range(len(self.process_groups)):
                    validator.check_isinstance("process_groups[" + str(i) +"]", self.process_groups[i], list)
                    self.group_device_num = len(self.process_groups[i])
                    if self.rank_id in self.process_groups[i] and self.group_device_num > 1:
                        self.is_global = True
                        if SYNC_BN_GROUP_NAME == "":
                            SYNC_BN_GROUP_NAME = "sync_bn_group" + str(i)
                            management.create_group(SYNC_BN_GROUP_NAME, self.process_groups[i])
            elif self.rank_size > 1:
                self.is_global = True
                self.group_device_num = self.rank_size
                self.device_list = [i for i in range(0, self.rank_size)]
                if SYNC_BN_GROUP_NAME == "":
                    SYNC_BN_GROUP_NAME = "sync_bn_group0"
                    management.create_group(SYNC_BN_GROUP_NAME, self.device_list)

        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reshape = P.Reshape()
        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.momentum = 1.0 - decay
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
        else:
            self.is_ge_backend = False

        self.bn_train = P.BatchNorm(is_training=True,
                                    epsilon=self.eps,
                                    momentum=self.momentum,
                                    data_format=self.format)
        if self.is_global:
            self.bn_train = inner.SyncBatchNorm(epsilon=self.eps,
                                                momentum=self.momentum,
                                                group=SYNC_BN_GROUP_NAME,
                                                device_num=self.group_device_num)

        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps, data_format=self.format)

        data_parallel_strategy = ((1,), (1,))
        data_parallel_strategy_one = ((1,), ())
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)

    def list_group(self, world_rank, group_size):
        if group_size > get_group_size():
            raise ValueError("group size can not be greater than local rank size, group size is {}, "
                             "local_rank_size is {}".format(group_size, get_group_size()))
        if len(world_rank) % group_size != 0:
            raise ValueError("please make your group size correct.")
        world_rank_list = zip(*(iter(world_rank),) * group_size)
        group_list = [list(i) for i in world_rank_list]
        return group_list

    def _check_rank_ids(self, process_groups, rank_size):
        seen = set()
        for rid in itertools.chain(*process_groups):
            validator.check_int_range(rid, 0, rank_size, Rel.INC_LEFT, "rank id in process_groups")
            if rid in seen:
                raise ValueError("rank id in process_groups should not be duplicated.")
            seen.add(rid)

    def construct(self, inputs):
        x_shape = F.shape(inputs)
        if len(x_shape) == 5:
            inputs = self.reshape(inputs, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))

        flag = self.use_batch_statistics

        if flag:
            output = self.bn_train(inputs,
                                 self.gamma,
                                 self.beta,
                                 self.moving_mean,
                                 self.moving_variance)[0]

            if len(x_shape) == 5:
                output = self.reshape(output, x_shape)
            return output

        output = self.bn_infer(inputs,
                             self.gamma,
                             self.beta,
                             self.moving_mean,
                             self.moving_variance)[0]
        if len(x_shape) == 5:
            output = self.reshape(output, x_shape)
        return output

    def extend_repr(self):
        return 'num_features={}, eps={}, momentum={}, gamma={}, beta={}, moving_mean={}, moving_variance={}'.format(
            self.num_features, self.eps, self.momentum, self.gamma, self.beta, self.moving_mean, self.moving_variance)


class GroupConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, groups):
        super(GroupConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]

        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=groups, data_format=self.data_format
        )

    def construct(self, inputs, filters):
        outputs = self.conv2d(inputs, filters)
        return outputs


class SeparableConv1D(Cell):

    def __init__(self, stride, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        super(SeparableConv1D, self).__init__()
        self.data_format, self.padding = preprocess_1d_format(data_format, padding)
        self.stride = (1, stride)
        self.dilations = (1, dilations)
        self.k_size = (1, k_size)
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.depth_multiplier = depth_multiplier
        self.depthwise_conv = P.Conv2D(
            out_channel=self.in_channel * self.depth_multiplier, kernel_size=self.k_size, pad_mode=self.padding,
            stride=self.stride, dilation=self.dilations, mode=1, group=self.in_channel
        )

        self.pointwise_conv = P.Conv2D(
            out_channel=self.out_channel, kernel_size=(1, 1), pad_mode=self.padding, stride=(1, 1), dilation=(1, 1),
            mode=1, group=1
        )

        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, x, depthwise_filters, pointwise_filters):

        if self.data_format == 'NWC':
            x = nhwc_to_nchw(x)

        x = self.expand_dims(x, 2)
        depthwise_filters = self.expand_dims(depthwise_filters, 2)
        pointwise_filters = self.expand_dims(pointwise_filters, 2)

        outputs = self.depthwise_conv(x, depthwise_filters)
        outputs = self.pointwise_conv(outputs, pointwise_filters)

        outputs = self.squeeze(outputs)

        if self.data_format == 'NWC':
            outputs = nchw_to_nhwc(outputs)
        return outputs


class SeparableConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel, depth_multiplier):
        super(SeparableConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.k_size = k_size
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.depth_multiplier = depth_multiplier

        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.depthwise_conv = P.Conv2D(
            out_channel=self.in_channel * self.depth_multiplier, kernel_size=self.k_size, pad_mode=self.padding,
            stride=self.ms_stride, dilation=self.ms_dilation, mode=1, group=self.in_channel , data_format=self.data_format
        )

        self.pointwise_conv = P.Conv2D(
            out_channel=self.out_channel, kernel_size=(1, 1), pad_mode=self.padding, stride=(1, 1), dilation=(1, 1),
            mode=1, group=1 , data_format=self.data_format
        )

    def construct(self, x, depthwise_filters, pointwise_filters):
        outputs = self.depthwise_conv(x, depthwise_filters)
        outputs = self.pointwise_conv(outputs, pointwise_filters)
        return outputs


class AdaptiveMeanPool1D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMeanPool1D, self).__init__()
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, inputs):

        if self.data_format == 'NWC':
            n, w, c = inputs.shape
            inputs = nhwc_to_nchw(inputs)
        else:
            n, c, w = inputs.shape
        inputs = self.expand_dims(inputs, 2)

        stride = (1, w // self.output_size)
        kernel = (1, w - (self.output_size - 1) * stride[1])
        outputs = P.AvgPool(kernel_size=kernel, strides=stride, pad_mode='VALID')(inputs)
        outputs = self.squeeze(outputs)

        if self.data_format == 'NWC':
            outputs = nchw_to_nhwc(outputs)

        return outputs


class AdaptiveMeanPool2D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMeanPool2D, self).__init__()
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def construct(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
            inputs = nhwc_to_nchw(inputs)
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = h // out_h
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = w // out_w
        kernel_w = w - (out_w - 1) * stride_w
        outputs = P.AvgPool(kernel_size=(kernel_h, kernel_w), strides=(stride_h, stride_w), pad_mode='VALID')(inputs)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)

        return outputs


class AdaptiveMeanPool3D(Cell):

    pass


class AdaptiveMaxPool1D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMaxPool1D, self).__init__()
        self.data_format, _ = preprocess_1d_format(data_format, None)
        self.output_size = output_size
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def construct(self, inputs):

        if self.data_format == 'NWC':
            n, w, c = inputs.shape
            inputs = nhwc_to_nchw(inputs)
        else:
            n, c, w = inputs.shape
        inputs = self.expand_dims(inputs, 2)

        stride = (1, w // self.output_size)
        kernel = (1, w - (self.output_size - 1) * stride[1])
        outputs = P.MaxPool(kernel_size=kernel, strides=stride, pad_mode='VALID')(inputs)
        outputs = self.squeeze(outputs)

        if self.data_format == 'NWC':
            outputs = nchw_to_nhwc(outputs)

        return outputs


class AdaptiveMaxPool2D(Cell):

    def __init__(self, output_size, data_format):
        super(AdaptiveMaxPool2D, self).__init__()
        self.data_format, _ = preprocess_2d_format(data_format, None)
        self.output_size = output_size

    def construct(self, inputs):

        if self.data_format == 'NHWC':
            n, h, w, c = inputs.shape
            inputs = nhwc_to_nchw(inputs)
        else:
            n, c, h, w = inputs.shape

        out_h, out_w = self.output_size
        stride_h = h // out_h
        kernel_h = h - (out_h - 1) * stride_h
        stride_w = w // out_w
        kernel_w = w - (out_w - 1) * stride_w
        outputs = P.MaxPool(kernel_size=(kernel_h, kernel_w), strides=(stride_h, stride_w),
                            pad_mode='VALID', data_format=self.data_format)(inputs)

        return outputs


class AdaptiveMaxPool3D(Cell):

    pass


class BinaryConv2D(Cell):

    def __init__(self, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        super(BinaryConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1, data_format=self.data_format
        )

        @bprop_getters.register(P.Sign)
        def get_bprop_Sign(self):

            def bprop(x, out, dout):

                grad = P.clip_by_value(dout, -1, 1)
                return (grad, )

            return bprop

        self.sign = P.Sign()

    def construct(self, inputs, filters):

        filters = self.sign(filters)
        outputs = self.conv2d(inputs, filters)

        return outputs


class DorefaConv2D(Cell):

    def __init__(self, bitW, bitA, strides, padding, data_format, dilations, out_channel, k_size, in_channel):
        super(DorefaConv2D, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format, padding)
        self.bitW = ms.Tensor(bitW)
        self.bitA = ms.Tensor(bitA)
        if self.data_format is 'NHWC':
            self.ms_stride = strides[1]
            self.ms_dilation = dilations[1]
            # self.transpose = P.Transpose()
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1
        )

        @bprop_getters.register(P.Round)
        def get_bprop_Round(self):

            def bprop(x, out, dout):

                return (dout, )

            return bprop

        @bprop_getters.register(P.Sign)
        def get_bprop_Sign(self):

            def bprop(x, out, dout):

                return (dout, )

            return bprop

        self.mimimum = P.Minimum()
        self.abs = P.Abs()
        self.round = P.Round()
        self.reducemean = P.ReduceMean()
        self.sign = P.Sign()
        self.pow = P.Pow()
        self.sub = P.Sub()
        self.oneslike = P.OnesLike()

    def cabs(self, inputs):

        a = P.stop_gradient(self.oneslike(inputs))
        return self.mimimum(self.abs(inputs), a)

    def _quantize_dorefa(self, x, k):

        n = self.sub(self.pow(2.0, k), 1)
        return self.round(x * n) / n

    def quantize_active(self, x, bitA):
        if bitA == 32:
            return x
        return self._quantize_dorefa(x, bitA)

    def quantize_weight(self, x, bitW, force_quantization=False):

        if bitW == 32 and not force_quantization:
            return x

        if bitW == 1:
            E = P.stop_gradient(self.reducemean(self.abs(x)))
            return self.sign(x / E) * E

        x = P.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)

        return 2 * self._quantize_dorefa(x, bitW) - 1

    def construct(self, inputs, filters):

        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        inputs = self.quantize_active(self.cabs(inputs), self.bitA)

        filters = self.quantize_weight(filters, self.bitW)

        outputs = self.conv2d(inputs, filters)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)

        return outputs
