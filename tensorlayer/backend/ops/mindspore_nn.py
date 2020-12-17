#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from mindspore.nn.cell import Cell
from mindspore import context
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.communication.management import get_group_size, get_rank
from mindspore.communication import management
from mindspore._checkparam import check_int_positive
from mindspore._extends import cell_attr_register


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


def leaky_relu(x):
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

    pass


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

    pass


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

    pass


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
    pass
    # raise NotImplementedError


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
            # self.transpose = P.Transpose()
        elif self.data_format is 'NCHW':
            self.ms_stride = strides[2]
            self.ms_dilation = dilations[2]

        # print(out_channel, k_size, self.padding, self.ms_stride, self.ms_dilation)
        self.conv2d = P.Conv2D(
            out_channel=out_channel, kernel_size=k_size, pad_mode=self.padding, stride=self.ms_stride,
            dilation=self.ms_dilation, mode=1, group=1
        )

    def construct(self, inputs, filters):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        outputs = self.conv2d(inputs, filters)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)
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
    pass
    # raise NotImplementedError


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


class MaxPool(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(MaxPool, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format=data_format, padding=padding)
        ms_ksize = ksize[1]
        ms_strides = strides[1]
        self.maxpool = P.MaxPool(ksize=ms_ksize, strides=ms_strides, padding=self.padding)

    def construct(self, inputs):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        outputs = self.maxpool(inputs)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)
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

    Returns
    -------
        A Tensor of format specified by data_format. The max pooled output tensor.
    """
    data_format, padding = preprocess_2d_format(data_format=data_format, padding=padding)
    if data_format == 'NHWC':
        input = nhwc_to_nchw(input)

    ms_ksize = ksize[1]
    ms_strides = strides[2]
    outputs = P.MaxPool(ksize=ms_ksize, strides=ms_strides, padding=padding)(input)
    # channel first to channel last
    if data_format == 'NHWC':
        outputs = nchw_to_nhwc(outputs)
    return outputs


class AvgPool(Cell):

    def __init__(self, ksize, strides, padding, data_format=None):
        super(AvgPool, self).__init__()
        self.data_format, self.padding = preprocess_2d_format(data_format=data_format, padding=padding)
        ms_ksize = ksize[1]
        ms_strides = strides[1]
        self.avgpool = P.AvgPool(ksize=ms_ksize, strides=ms_strides, padding=padding)

    def construct(self, inputs):
        if self.data_format == 'NHWC':
            inputs = nhwc_to_nchw(inputs)

        outputs = self.avgpool(inputs)

        if self.data_format == 'NHWC':
            outputs = nchw_to_nhwc(outputs)
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
    def __init__(
        self, num_features, epsilon=1e-5, decay=0.9, gamma=None, beta=None, moving_mean=None, moving_var=None,
        is_train=None, device_num_each_group=1, data_format='channels_last'
    ):
        super(BatchNorm, self).__init__()
        if num_features < 1:
            raise ValueError("num_features must be at least 1")

        if decay < 0 or decay > 1:
            raise ValueError("momentum should be a number in range [0, 1], but got {}".format(decay))

        self.data_format = data_format
        self.use_batch_statistics = is_train
        self.num_features = num_features
        self.eps = epsilon
        self.moving_mean = moving_mean
        self.moving_variance = moving_var
        self.gamma = gamma
        self.beta = beta
        self.group = check_int_positive(device_num_each_group)
        self.is_global = False
        if self.group != 1:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()
            self.device_list = [i for i in range(0, self.rank_size)]
            self.rank_list = self.list_group(self.device_list, self.group)
            self.rank_list_idx = len(self.rank_list)
            for i in range(self.rank_list_idx):
                if self.rank_id in self.rank_list[i] and self.group != 1:
                    self.is_global = True
                    management.create_group('group' + str(i), self.rank_list[i])
                    self.all_reduce = P.AllReduce(P.ReduceOp.SUM, 'group' + str(i)).add_prim_attr('fusion', 1)
        self.shape = P.Shape()
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reshape = P.Reshape()
        self.is_ascend = context.get_context("device_target") == "Ascend"
        self.is_gpu = context.get_context("device_target") == "GPU"
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.momentum = 1.0 - decay
        if context.get_context("enable_ge"):
            self.is_ge_backend = True
        else:
            self.is_ge_backend = False

        if self.is_graph_mode and (self.is_ge_backend or self.is_ascend):
            self.bn_train = P.BatchNorm(is_training=True, epsilon=self.eps)
        elif self.is_gpu:
            self.bn_train = P.FusedBatchNormEx(mode=1, epsilon=self.eps, momentum=self.momentum)
        else:
            self.bn_train = P.FusedBatchNorm(mode=1, epsilon=self.eps, momentum=self.momentum)
        self.bn_infer = P.BatchNorm(is_training=False, epsilon=self.eps)
        self.enable_global_sync = self.is_global and (self.is_ge_backend or (self.is_graph_mode and self.is_ascend))
        self.enable_default_train = self.is_graph_mode and not self.is_global and \
                                    (self.is_ge_backend or self.is_ascend)

        data_parallel_strategy = ((1, ), (1, ))
        data_parallel_strategy_one = ((1, ), ())
        self.sub_mean = P.Sub().shard(data_parallel_strategy)
        self.sub_var = P.Sub().shard(data_parallel_strategy)
        self.mul_mean = P.Mul().shard(data_parallel_strategy_one)
        self.mul_var = P.Mul().shard(data_parallel_strategy_one)
        self.assign_sub_mean = P.AssignSub().shard(data_parallel_strategy)
        self.assign_sub_var = P.AssignSub().shard(data_parallel_strategy)

    def _check_data_dim(self, x):
        raise NotImplementedError

    def list_group(self, world_rank, group_size):
        if group_size > get_group_size():
            raise ValueError(
                "group size can not be greater than local rank size, group size is {}, "
                "local_rank_size is {}".format(group_size, get_group_size())
            )
        if len(world_rank) % group_size != 0:
            raise ValueError("please make your group size correct.")
        world_rank_list = zip(*(iter(world_rank), ) * group_size)
        group_list = [list(i) for i in world_rank_list]
        return group_list

    def _global_sync(self, x, axes, re_shape):
        """calculate global batch normalization output"""
        x_mean = self.reduce_mean(x, axes)
        x_mean_square = self.reduce_mean(self.square(x), axes)
        global_batch_mean = self.all_reduce(x_mean) / self.group
        global_batch_mean_square = self.all_reduce(x_mean_square) / self.group
        global_mean = global_batch_mean
        global_var = global_batch_mean_square - self.square(global_mean)
        var_sqrt = self.sqrt(global_var + self.eps)
        mean_first = (x - global_mean) / var_sqrt
        y = mean_first * self.reshape(self.gamma, re_shape) + self.reshape(self.beta, re_shape)

        mean_sub = self.sub_mean(self.reshape(self.moving_mean, re_shape), global_mean)
        tmp_mean = self.mul_mean(mean_sub, self.cast(self.momentum, self.dtype(mean_sub)))
        mean_sub2 = self.sub_var(self.reshape(self.moving_mean, re_shape), global_var)
        tmp_variance = self.mul_var(mean_sub2, self.cast(self.momentum, self.dtype(mean_sub2)))
        y = F.depend(y, self.assign_sub_mean(self.moving_mean, self.reshape(tmp_mean, self.shape(self.moving_mean))))
        y = F.depend(
            y, self.assign_sub_var(self.moving_variance, self.reshape(tmp_variance, self.shape(self.moving_variance)))
        )
        return y

    def get_dim(self, input):
        dim = len(self.shape(input))
        if dim == 2:
            return '1d'
        elif dim == 4:
            return '2d'
        else:
            raise ValueError("The input must has 2 dims or 4 dims.")

    def _shape_check_bn(self, in_shape, in_dims):
        dim = len(in_shape)
        if in_dims == '1d' and dim != 2:
            raise ValueError("The input must has 2 dims.")
        if in_dims == '2d' and dim != 4:
            raise ValueError("The input must has 4 dims.")
        if in_dims == 'both' and dim != 2 and dim != 4:
            raise ValueError("The input must has 2 dims or 4 dims.")

    def _shape_infer(self, x_shape, num_feature):
        """global batch normalization shape and axes infer"""
        if len(x_shape) == 4:
            axes = (0, 2, 3)
            re_shape = (1, num_feature, 1, 1)
        else:
            axes = (0, )
            re_shape = (1, num_feature)
        return axes, re_shape

    def construct(self, inputs):
        x = inputs
        self._shape_check_bn(self.shape(x), self.get_dim(x))
        if self.use_batch_statistics is None:
            flag = self.training
        else:
            flag = self.use_batch_statistics

        if flag:
            if self.enable_global_sync:
                if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                    x = nhwc_to_nchw(x)
                axes, re_shape = self._shape_infer(F.shape(x), self.num_features)
                y = self._global_sync(x, axes, re_shape)
                if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                    y = nchw_to_nhwc(y)
                return y

            if self.enable_default_train:
                if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                    x = nhwc_to_nchw(x)
                y, batch_mean, batch_var, _, _ = self.bn_train(x, self.gamma, self.beta, None, None)

                mean_sub = self.sub_mean(self.moving_mean, batch_mean)
                temp_mean = self.mul_mean(mean_sub, self.momentum)
                mean_sub2 = self.sub_var(self.moving_variance, batch_var)
                temp_variance = self.mul_var(mean_sub2, self.momentum)
                y = F.depend(y, self.assign_sub_mean(self.moving_mean, temp_mean))
                y = F.depend(y, self.assign_sub_var(self.moving_variance, temp_variance))
                if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                    y = nchw_to_nhwc(y)
                return y

            if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                x = nhwc_to_nchw(x)
            y = self.bn_train(x, self.gamma, self.beta, self.moving_mean, self.moving_variance)[0]
            if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
                y = nchw_to_nhwc(y)
            return y
        if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
            x = nhwc_to_nchw(x)
        y = self.bn_infer(x, self.gamma, self.beta, self.moving_mean, self.moving_variance)[0]
        if self.data_format == 'channels_last' and self.get_dim(x) == '2d':
            y = nchw_to_nhwc(y)
        return y
