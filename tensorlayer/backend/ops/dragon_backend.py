#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import dragon as D

from dragon.core.eager import context
from dragon.core.ops import init_ops
from dragon.core.ops import vision_ops

_dtypeDict = ['float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
# TODO NotImplemented
DType = None
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
uint8 = 'uint8'
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'

# isinstance input output
# TODO NotImplemented
# TensorLike = None


def _getter(init_fn, **kwargs):
    """Return an named eager tensor."""
    with context.eager_mode():
        value = init_fn(**kwargs)
        value._name = kwargs.get('name', value.id)
    return value


def set_context(**kwargs):
    raise Exception("Using Dragon backend,You don't need to set context")


def get_tensor_shape(x):
    return x.shape


# initializers
def zeros(shape, dtype='float32'):
    """
    Creates a tensor with all elements set to zero.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to zero.

    """
    return _getter(
        init_ops.fill,
        value=0,
        shape=shape,
        dtype=dtype,
    )


def ones(shape, dtype='float32'):
    """
    Creates a tensor with all elements set to ones.

    Parameters
    ----------
    shape : A list of integers
        a tuple of integers, or a 1-D Tensor of type int32.
    dtype : tensor
        The DType of an element in the resulting Tensor

    Returns
    -------
        A Tensor with all elements set to zero.

    """
    return _getter(
        init_ops.fill,
        value=1,
        shape=shape,
        dtype=dtype,
    )


def constant(value, shape, dtype='float32'):
    """
    Creates a constant tensor from a tensor-like object.

    Parameters
    ----------
    value : list
        A constant value (or list) of output type dtype.
    dtype : tensor
         The type of the elements of the resulting tensor.
    shape : tuple
        Optional dimensions of resulting tensor.

    Returns
    -------
        A Constant Tensor.

    """
    # shape = shape[::-1]
    return _getter(
        init_ops.fill,
        value=value,
        shape=shape,
        dtype=dtype,
    )


def random_uniform(shape, minval=0, maxval=None, dtype='float32', seed=None):
    """
    Outputs random values from a uniform distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    minval : int
        The lower bound on the range of random values to generate (inclusive). Defaults to 0.
    maxval : int
        The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
    dtype : tensor
        The type of the output: float16, float32, float64, int32, or int64.
    seed : int
         Used in combination with dragon.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """
    return _getter(init_ops.random_uniform, low=minval, high=maxval, shape=shape, dtype=dtype)


def random_normal(shape, mean=0.0, stddev=1.0, dtype='float32', seed=None):
    """
    Outputs random values from a normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random normal values.

    """
    return _getter(
        init_ops.random_normal,
        mean=mean,
        std=stddev,
        shape=shape,
        dtype=dtype,
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype='float32', seed=None):
    """
    Outputs random values from a truncated normal distribution.

    Parameters
    ----------
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    dtype : tensor
        The type of the output.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns
    -------
        A tensor of the specified shape filled with random truncated normal values.

    """
    return _getter(
        init_ops.truncated_normal,
        mean=mean,
        std=stddev,
        shape=shape,
        dtype=dtype,
    )


def he_normal(shape, dtype, seed=None):
    """
    He normal initializer.

    Parameters
    ----------
    seed : A Python integer.
        Used to seed the random generator.
    shape : tuple
        A 1-D integer Tensor or Python array. The shape of the output tensor.
    dtype : tensor
        The type of the output.

    Returns
    -------
        A tensor of the specified shape filled with he normal values.
    """
    # shape = shape[::-1]
    raise NotImplementedError("He_Normal is not implemented")


def Variable(initial_value, name, trainable=None):
    """
    Creates a new variable with value initial_value.

    Parameters
    ----------
    initial_value : tensor
        A Tensor, or Python object convertible to a Tensor
    name : str
        Optional name for the variable. Defaults to 'Variable' and gets uniquified automatically.
    Returns
    -------
        Variable
    """
    return D.Tensor(name=name, shape=initial_value)


class MatMul(object):

    def __init__(self):
        pass

    def __call__(self, a, b):
        inputs = [a, b]
        return D.math.matmul(inputs)


def matmul(a, b):
    """
    Multiplies matrix a by matrix b, producing a * b.

    Parameters
    ----------
    a : tensor
         type float16, float32, float64, int32, complex64, complex128 and rank > 1.
    b : tensor
        with same type and rank as a.

    Returns
    -------
        A Tensor of the same type as a and b
    """
    inputs = [a, b]
    return D.math.matmul(inputs)


def add(value, bias):
    """
    Returns x + y element-wise.

    Parameters
    ----------
    value :  tensor.
        Must be one of the following types: bfloat16, half, float32, float64,
        uint8, int8, int16, int32, int64, complex64, complex128, string.
    bias : tensor
        Must have the same type as a
    name : str
        A name for the operation

    Returns
    -------
        A Tensor. Has the same type as a.
    """

    inputs = [value, bias]
    return D.math.add(inputs)


def dtypes(dt):
    """
    Data dtypes.

    Parameters
    ----------
    dt : string
         It could be 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16',
         'int32', 'int64', 'float16', 'float32', 'float64', 'DType'.

    Returns
    -------
        Data dtypes
    """
    if dt not in _dtypeDict:
        raise Exception("Unsupported dtype: {}".format(dt))
    return dt


def minimum(x, y):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Parameters
    ----------
    x : tensor.
        Must be one of the following types: bfloat16, half, float32, float64, int32, int64.
    y : A Tensor.
        Must have the same type as x.
    name : str
        A name for the operation (optional).

    Returns
    -------
        A Tensor. Has the same type as x
    """
    inputs = [x, y]
    return D.math.minimum(inputs)


class FlattenReshape(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        dim = 1
        for d in get_tensor_shape(inputs)[1:]:
            dim *= d
        return D.reshape(inputs, [-1, dim])


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, tensor):
        return D.reshape(tensor, shape=self.shape)


def reshape(tensor, shape):
    """
    Reshapes a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    shape : tensor
         Defines the shape of the output tensor.
    Returns
    -------
        A Tensor. Has the same type as tensor
    """
    return D.reshape(tensor, shape=shape)


class Concat(object):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __call__(self, values):
        return D.concat(values=values, axis=self.axis)


def concat(values, axis):
    """
    Concatenates tensors along one dimension.

    Parameters
    ----------
    values : list
         A list of Tensor objects or a single Tensor
    axis : int
        0-D int32 Tensor. Dimension along which to concatenate
    Returns
    -------
        A Tensor resulting from concatenation of the input tensors.
    """
    return D.concat(values, axis=axis)


def convert_to_tensor(value, dtype=None):
    """
    Converts the given value to a Tensor.

    Parameters
    ----------
    value : object
        An object whose type has a registered Tensor conversion function.
    dtype : optional
        Optional element type for the returned tensor. If missing, the type is inferred from the type of value.

    Returns
    -------
        A Tensor based on value.
    """
    return D.Tensor.convert_to(value, dtype)


def sqrt(x):
    """
    Computes square root of x element-wise.

    Parameters
    ----------
    x : tensor
         Must be one of the following types: bfloat16, half, float32, float64, complex64, complex128.

    Returns
    -------
        A Tensor. Has the same type as x.
    """
    return D.math.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis):
        pass

    def construct(self, input):
        pass


class ReduceMean(object):

    def __init__(self, axis):
        if axis == [1, 2]:
            self.data_format = 'NHWC'
        elif axis == [2, 3]:
            self.data_format = 'NCHW'
        else:
            raise ("`data_format` should have one of the following values: [`channels_last`, `channels_first`]")

    def __call__(self, inputs):
        return vision_ops.pool2d(
            inputs,
            kernel_shape=1,
            strides=1,
            pads=0,
            mode='AVG',
            global_pooling=True,
            data_format=self.data_format,
        )


def reduce_mean(input_tensor, axis=None):
    """
    Computes the mean of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    return D.mean(input_tensor, axes=axis)


class ReduceMax(object):

    def __init__(self, axis):
        if axis == [1, 2]:
            self.data_format = 'NHWC'
        elif axis == [2, 3]:
            self.data_format = 'NCHW'
        else:
            raise ("`data_format` should have one of the following values: [`channels_last`, `channels_first`]")

    def __call__(self, inputs):
        return vision_ops.pool2d(
            inputs, kernel_shape=1, strides=1, pads=0, mode='MAX', global_pooling=True, data_format=self.data_format
        )


def reduce_max(input_tensor, axis=None):
    """
    Computes the maximum of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """

    return D.max(input_tensor, axis)


def reduce_min(input_tensor, axis=None):
    """
    Computes the minimum of elements across dimensions of a tensor.

    Parameters
    ----------
    input_tensor : tensor
        The tensor to reduce. Should have real numeric type.
    axis : int
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(input_tensor), rank(input_tensor)).
    name : str
        A name for the operation (optional).

    Returns
    -------
        The reduced tensor.
    """
    return D.min(input_tensor, axis)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0):
    """
    Pads a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    paddings : tuple
        A tuple of type int32.
    mode : str
        One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    constant_values : int
        In "CONSTANT" mode, the scalar pad value to use. Must be same type as tensor.

    Returns
    -------
        A Tensor. Has the same type as tensor.
    """
    if mode not in ['CONSTANT', 'REFLECT', 'SYMMETRIC']:
        raise Exception("Unsupported mode: {}".format(mode))
    if mode == 'SYMMETRIC':
        mode = 'EDGE'
    outputs = D.pad(tensor, pads=paddings, mode=mode, value=constant_values)
    return outputs


class Unstack(object):

    def __init__(self, axis, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        raise NotImplementedError


class Stack(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, values):
        return D.stack(values, axis=self.axis)


def stack(values, axis=0):
    """
    Stacks a list of rank-R tensors into one rank-(R+1) tensor.

    Parameters
    ----------
    values : list
        A list of Tensor objects with the same shape and type.
    axis : int
        An int. The axis to stack along. Defaults to the first dimension.
        Negative values wrap around, so the valid range is [-(R+1), R+1).

    Returns
    -------
        A stacked Tensor with the same type as values.
    """
    return D.stack(values, axis=axis)


def meshgrid(x, y):
    """
    Broadcasts parameters for evaluation on an N-D grid.

    Parameters
    ----------
    x : tensor
        Tensors with rank 1.
    y : tensor
        Tensors with rank 1.

    Returns
    -------
        A list of N Tensors with rank N.
    """

    pass


def range(start, limit=None, delta=1, dtype=None):
    """
    Creates a sequence of numbers.

    Parameters
    ----------
    start : tensor
        A 0-D Tensor (scalar). Acts as first entry in the range if limit is not None;
        otherwise, acts as range limit and first entry defaults to 0.
    limit : tensor
         A 0-D Tensor (scalar). Upper limit of sequence, exclusive. If None,
         defaults to the value of start while the first entry of the range defaults to 0.
    delta : tensor
        A 0-D Tensor (scalar). Number that increments start. Defaults to 1.
    dtype : type
        The type of the elements of the resulting tensor.

    Returns
    -------
        An 1-D Tensor of type dtype.
    """
    if dtype is None:
        dtype = 'int32'
    if limit is None:
        outputs = D.arange(start=0, stop=start, step=delta, dtype=dtype)
    else:
        outputs = D.arange(start, stop=limit, step=delta, dtype=dtype)
    return outputs


class ExpandDims(object):

    def __init__(self, axis):
        pass

    def construct(self, input):
        pass


def expand_dims(input, axis):
    """
    Inserts a dimension of 1 into a tensor's shape.

    Parameters
    ----------
    input : tensor
        A Tensor.
    axis : int
        0-D (scalar). Specifies the dimension index at which to expand the shape of input.
        Must be in the range [-rank(input) - 1, rank(input)].

    Returns
    -------
        A Tensor with the same data as input, but its shape has an additional dimension of size 1 added.
    """

    return D.expand_dims(input, axis=axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return D.tile(input, multiples)


def tile(input, multiples):
    """
    Constructs a tensor by tiling a given tensor.

    Parameters
    ----------
    input : tensor
        A Tensor. 1-D or higher.
    multiples : tensor
        Must be one of the following types: int32, int64. 1-D.
        Length must be the same as the number of dimensions in input

    Returns
    -------
        A Tensor. Has the same type as input.
    """
    return D.tile(input, multiples)


class Cast(object):

    def __init__(self, dtype):
        pass

    def __call__(self, input):
        pass


def cast(x, dtype):
    """
    Casts a tensor to a new type.

    Parameters
    ----------
    x : tensor
        A Tensor or SparseTensor or IndexedSlices of numeric type.
        It could be uint8, uint16, uint32, uint64, int8, int16, int32, int64, float16, float32, float64.
    dtype : dtpye
         The destination type. The list of supported dtypes is the same as x

    Returns
    -------
        A Tensor or SparseTensor or IndexedSlices with same shape as x and same type as dtype.
    """
    return D.cast(x, dtype=dtype)


class Transpose(object):

    def __init__(self, perm, conjugate=False):
        self.perm = perm
        if conjugate:
            raise ("The conjugate Parameters not supported")

    def __call__(self, a):
        return D.transpose(a, self.perm)


def transpose(a, perm=None, conjugate=False):
    """
    Transposes a.

    Parameters
    ----------
    a : tensor
        A Tensor.
    perm : int
        A permutation of the dimensions of a.
    conjugate : bool
        Setting it to True is mathematically equivalent to ms.math.conj(ms.transpose(input)).

    Returns
    -------
        A transposed Tensor.
    """

    conjugate = conjugate
    return D.transpose(a, perm=perm)


def gather_nd(params, indices, batch_dims=0):
    """
    Gather slices from params into a Tensor with shape specified by indices.

    Parameters
    ----------
    params : tensor
        The tensor from which to gather values.
    indices : tensor
        Must be one of the following types: int32, int64. Index tensor.
    batch_dims : int
        An integer or a scalar 'Tensor'. The number of batch dimensions.

    Returns
    -------
        A Tensor. Has the same type as params.
    """

    pass


def clip_by_value(t, clip_value_min, clip_value_max):
    """
    Clips tensor values to a specified min and max.

    Parameters
    ----------
    t : tensor
        A Tensor or IndexedSlices
    clip_value_min : tensor
        A 0-D (scalar) Tensor, or a Tensor with the same shape as t. The minimum value to clip by
    clip_value_max : tensor
        A 0-D (scalar) Tensor, or a Tensor with the same shape as t. The minimum value to clip by

    Returns
    -------
        A clipped Tensor or IndexedSlices.
    """

    pass


def split(value, num_or_size_splits, axis=0, num=None):
    """
    Splits a tensor into sub tensors.

    Parameters
    ----------
    value : tensor
        The Tensor to split.
    num_or_size_splits : list
        Either an integer indicating the number of splits along split_dim or a 1-D integer Tensor or
        Python list containing the sizes of each output tensor along split_dim.
    axis : int
        The dimension along which to split. Must be in the range [-rank(value), rank(value)). Defaults to 0.
    num : int
        used to specify the number of outputs when it cannot be inferred from the shape of size_splits.

    Returns
    -------
        Tensor objects resulting from splitting value.
    """
    pass


def floor(x):
    return D.math.floor(x)


def gather(params, indices):
    return NotImplementedError


def linspace(start, stop, num):
    return D.linspace(start, stop, num)


def slice(inputs, starts, sizes):
    return D.slice(inputs, starts, sizes)


def add_n(inputs):
    return NotImplementedError


class OneHot(object):

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype='float32'):
        self.depth = depth
        self.dtype = dtype

    def __call__(self, indices):
        outputs = np.zeros(shape=(indices.shape[0], self.depth))
        for i in np.arange(indices.shape[0]):
            outputs[int(i)][int(indices[int(i)].get_value())] = 1
        outputs = D.constant(outputs, dtype=self.dtype)
        return outputs


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        super(L2Normalize, self).__init__()
        pass

    def __call__(self, input, *args, **kwargs):
        pass


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm

    def __call__(self, params, ids, *args, **kwargs):
        pass


class NCELoss(object):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        super(NCELoss, self).__init__()

    def __call__(self, weights, biases, labels, inputs, num_sampled, num_classes):
        pass


class Not_equal(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        pass


class Count_nonzero(object):

    def __init__(self, keepdims=None, dtype='int64'):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last', ksize=None):
        if method not in ['nearest', 'linear', 'bilinear']:
            raise ('Current resize does not support this method.')
        if method == 'bilinear':
            method = 'linear'
        self.method = method
        self.antialias = antialias
        self.scale = scale
        if data_format != 'channel_last':
            raise Exception("UpSampling2d resize_images only support channel_last")

    def __call__(self, inputs):
        output_size = (int(inputs.shape[1] * self.scale[0]), int(inputs.shape[2] * self.scale[1]))
        outputs = D.vision.resize(inputs, sizes=output_size, mode=self.method, align_corners=self.antialias)
        return outputs


def resize(inputs, output_size, method, antialias):
    if method not in ['nearest', 'linear', 'bilinear']:
        raise ('Current resize does not support this method.')
    if method == 'bilinear':
        method = 'linear'
    return D.vision.resize(inputs, sizes=output_size, mode=method, align_corners=antialias)


class ZeroPadding1D(object):

    def __init__(self):
        pass

    def __call__(self, padding):
        raise NotImplementedError


class ZeroPadding2D(object):

    def __init__(self):
        pass

    def __call__(self, padding):
        raise NotImplementedError


class ZeroPadding3D(object):

    def __init__(self):
        pass

    def __call__(self, padding):
        raise NotImplementedError


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return D.math.sign(x)
