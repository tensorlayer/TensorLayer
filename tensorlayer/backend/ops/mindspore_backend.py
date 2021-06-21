#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from .mindspore_nn import nchw_to_nhwc, nhwc_to_nchw
from mindspore._c_expression.typing import Type
from mindspore.common import dtype as mstype

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import (
    initializer, Constant, Normal, TruncatedNormal, Initializer, _assignment, _calculate_in_and_out, One, Zero
)
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore.context as context
from mindspore.nn import Cell
from mindspore.ops import count_nonzero
import mindspore.numpy as msnp

import numpy as np
from scipy.stats import truncnorm
import random

_dtypeDict = {
    'DType': Type,
    'float16': mstype.float16,
    'float32': mstype.float32,
    'float64': mstype.float64,
    'int8': mstype.int8,
    'int16': mstype.int16,
    'int32': mstype.int32,
    'int64': mstype.int64,
    'uint8': mstype.uint8,
    'uint16': mstype.uint16,
    'uint32': mstype.uint32,
    'uint64': mstype.uint64
}

DType = Type
float16 = mstype.float16
float32 = mstype.float32
float64 = mstype.float64
int8 = mstype.int8
int16 = mstype.int16
int32 = mstype.int32
int64 = mstype.int64
uint8 = mstype.uint8
uint16 = mstype.uint16
uint32 = mstype.uint32
uint64 = mstype.uint64

# isinstance input output
# TensorLike = Tensor_


def set_context(**kwargs):
    return context.set_context(**kwargs)


def get_tensor_shape(x):
    return list(P.Shape()(x))


# initializers
def zeros(shape, dtype=mstype.float32):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Zero()
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def ones(shape, dtype=mstype.float32):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = One()
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def constant(value, dtype=mstype.float32, shape=None):
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
    arr = np.ndarray(shape)
    Constant(value)(arr=arr)
    return Tensor(arr, dtype=dtype)


class Uniform(Initializer):
    """
    Initialize a uniform array, and obtain values U(-scale, scale) from the uniform distribution
    to fill the input tensor.

    Args:
        minval : int
        The lower bound on the range of random values to generate (inclusive). Defaults to 0.
        maxval : int
        The upper bound on the range of random values to generate (exclusive). Defaults to 1 if dtype is floating point.
        seed : int
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.

    Returns:
        Array, uniform array.
    """

    def __init__(self, minval=0, maxval=None, seed=None):
        super(Uniform, self).__init__(minval=minval, maxval=maxval, seed=seed)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def _initialize(self, arr):
        random.seed(self.seed)
        tmp = np.random.uniform(self.minval, self.maxval, arr.shape)
        _assignment(arr, tmp)


def random_uniform(shape, minval=0, maxval=None, dtype=mstype.float32, seed=None):
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
         Used in combination with tf.random.set_seed to create a reproducible sequence of tensors across multiple calls.
    Returns
    -------
        A tensor of the specified shape filled with random uniform values.

    """
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Uniform(minval=minval, maxval=maxval, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class Normal(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Parameters
    ----------
    mean : float
        The mean of the normal distribution
    stddev : float
        The standard deviation of the normal distribution.
    seed : A Python integer
         Used to create a random seed for the distribution

    Returns:
        Array, normal array.
    """

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(Normal, self).__init__(mean=mean, stddev=stddev)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def _initialize(self, arr):
        random.seed(self.seed)
        tmp = np.random.normal(self.mean, self.stddev, arr.shape)
        _assignment(arr, tmp)


class RandomNormal(Cell):

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(RandomNormal, self).__init__()
        self.normal = Normal(mean=mean, stddev=stddev, seed=seed)

    def construct(self, shape):
        arr = np.ndarray(shape)
        outputs = self.normal(arr)
        return outputs


def random_normal(shape, mean=0.0, stddev=1.0, dtype=mstype.float32, seed=None):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = Normal(mean=mean, stddev=stddev, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class TruncatedNormal(Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution within N(low, high).

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, truncated normal array.
    """

    def __init__(self, mean=0.0, stddev=0.01, seed=None):
        super(TruncatedNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def _initialize(self, arr):
        tmp = truncnorm.rvs(-2, 2, loc=self.mean, scale=self.stddev, size=arr.shape, random_state=None)
        _assignment(arr, tmp)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=mstype.float32, seed=None):
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
    # shape = shape[::-1]
    arr = np.ndarray(shape)
    init_obj = TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


class HeNormal(Initializer):
    r"""
    he_normal: It draws samples from a truncated normal distribution centered on 0 with
    stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.

    Args:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.
    """

    def __init__(self, seed=None):
        super(HeNormal, self).__init__(seed=seed)
        self.seed = seed

    def _initialize(self, arr):
        n_in, _ = _calculate_in_and_out(arr)
        boundary = np.sqrt(2.0 / n_in)
        random.seed(self.seed)
        data = np.random.normal(-boundary, boundary, arr.shape)
        _assignment(arr, data)


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
    arr = np.ndarray(shape)
    init_obj = HeNormal(seed)
    init_obj(arr)
    return Tensor(arr, dtype=dtype)


def Variable(initial_value, name, trainable=True):
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

    var = Parameter(initial_value, name=name, requires_grad=trainable)
    return var


class MatMul(Cell):

    def __init__(self):
        super(MatMul, self).__init__()
        self.matmul = P.MatMul()

    def construct(self, a, b):
        return self.matmul(a, b)


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
    matmul_obj = P.MatMul()
    outputs = matmul_obj(a, b)
    return outputs


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

    add_obj = P.TensorAdd()
    outputs = add_obj(value, bias)
    return outputs


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

    if dt not in _dtypeDict.keys():
        raise Exception("Unsupported dtype: {}".format(dt))
    return _dtypeDict[dt]


class Maximum(Cell):

    def __init__(self):
        super(Maximum, self).__init__()
        self.maximum = P.Maximum()

    def construct(self, x, y):
        return self.maximum(x, y)


class Minimum(Cell):

    def __init__(self):
        super(Minimum, self).__init__()
        self.minimum = P.Minimum()

    def construct(self, x, y):
        return self.minimum(x, y)


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
    minimum_obj = P.Minimum()
    outputs = minimum_obj(x, y)
    return outputs


class FlattenReshape(Cell):

    def __init__(self):
        super(FlattenReshape, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, inputs):
        dim = 1
        for d in self.shape(inputs)[1:]:
            dim *= d
        return self.reshape(inputs, (-1, dim))


class Reshape(Cell):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.reshape = P.Reshape()
        self.shape = tuple(shape)

    def construct(self, tensor):
        return self.reshape(tensor, self.shape)


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
    reshape_obj = P.Reshape()
    outputs = reshape_obj(tensor, tuple(shape))
    return outputs


class Concat(Cell):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.concat = P.Concat(axis)

    def construct(self, values):
        return self.concat(values)


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
    # TODO testing axis
    concat_obj = P.Concat(axis)
    outputs = concat_obj(values)
    return outputs


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
    #todo testing value
    return Tensor(value, dtype=dtype)


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
    sqrt_obj = P.Sqrt()
    outputs = sqrt_obj(x)
    return outputs


class ReduceSum(Cell):

    def __init__(self, axis):
        super(ReduceSum, self).__init__()
        self.axis = axis
        self.reduce_sum = P.ReduceSum(keep_dims=True)

    def construct(self, input):
        return self.reduce_sum(input, self.axis)


class ReduceMean(Cell):

    def __init__(self, axis):
        super(ReduceMean, self).__init__()
        self.axis = axis
        self.reducemean = P.ReduceMean(keep_dims=False)

    def construct(self, inputs):
        output = self.reducemean(inputs, self.axis)
        return output


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

    Rmean_obj = P.ReduceMean(keep_dims=False)
    outputs = Rmean_obj(input_tensor, axis)
    return outputs


class ReduceMax(Cell):

    def __init__(self, axis):
        super(ReduceMax, self).__init__()
        self.axis = axis
        self.reducemax = P.ReduceMax(keep_dims=False)

    def construct(self, inputs):
        output = self.reducemax(inputs, self.axis)
        return output


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

    Rmax_obj = P.ReduceMax(keep_dims=False)
    outputs = Rmax_obj(input_tensor, axis)
    return outputs


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

    Rmin_obj = P.ReduceMin(keep_dims=False)
    outputs = Rmin_obj(input_tensor, axis)
    return outputs


class Pad(Cell):

    def __init__(self, paddings, mode="REFLECT"):
        super(Pad, self).__init__()
        if mode not in ["REFLECT", "SYMMETRIC"]:
            raise Exception("Unsupported mode: {}".format(mode))
        self.pad = P.MirrorPad(mode=mode)
        self.paddings = Tensor(paddings)

    def construct(self, x):
        return self.pad(x, self.paddings)


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
    raise NotImplementedError


class Unstack(Cell):

    def __init__(self, axis, num=None):
        super(Unstack, self).__init__()
        if num is not None:
            raise ("The num Parameters do not need to be set.")
        self.unstack = P.Unpack(axis=axis)

    def construct(self, values):
        return self.unstack(values)


class Stack(Cell):

    def __init__(self, axis=0):
        super(Stack, self).__init__()
        self.stack = P.Pack(axis=axis)

    def construct(self, values):
        return self.stack(values)


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
    _stack = P.Pack(axis=axis)
    return _stack(values)


class Meshgrid(Cell):

    def __init__(self, indexing='xy'):
        super(Meshgrid, self).__init__()
        self._meshgrid = P.Meshgrid(indexing=indexing)

    def construct(self, *args):
        inputs = tuple(*args)
        return self._meshgrid(inputs)


def meshgrid(*args, **kwargs):
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

    _meshgrid = P.Meshgrid(**kwargs)
    return _meshgrid(*args)


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

    pass


class ExpandDims(Cell):

    def __init__(self, axis):
        super(ExpandDims, self).__init__()
        self.axis = axis
        self.expand_dims = P.ExpandDims()

    def construct(self, input):
        output = self.expand_dims(input, self.axis)
        return output


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

    expand_obj = P.ExpandDims()
    outputs = expand_obj(input, axis)
    return outputs


class Tile(Cell):

    def __init__(self):
        super(Tile, self).__init__()
        self.tile = P.Tile()

    def construct(self, input, multiples):
        return self.tile(input, tuple(multiples))


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
    tile_obj = P.Tile()
    outputs = tile_obj(input, multiples)
    return outputs


class Cast(Cell):

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self.dtype = dtype
        self.cast = P.Cast()

    def construct(self, input):
        return self.cast(input, self.dtype)


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
    cast_obj = P.Cast()
    outputs = cast_obj(x, dtype)
    return outputs


class Transpose(Cell):

    def __init__(self, perm, conjugate=False):
        super(Transpose, self).__init__()
        self.perm = tuple(perm)
        self.conjugate = conjugate
        self.transpose = P.Transpose()
        if self.conjugate:
            raise NotImplementedError("conjugate not implemented")

    def construct(self, a):
        return self.transpose(a, self.perm)


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
    # TODO conjugate
    trans_obj = P.Transpose()
    outputs = trans_obj(a, perm)
    print(outputs)


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
    min_value = Tensor(clip_value_min, mstype.float32)
    max_value = Tensor(clip_value_max, mstype.float32)
    output = C.clip_by_value(t, min_value, max_value)
    return output


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


class Floor(Cell):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def floor(x):
    return NotImplementedError


def gather(params, indices):
    return NotImplementedError


def linspace(start, stop, num):
    return NotImplementedError


def slice(inputs, starts, sizes):
    return NotImplementedError


def add_n(inputs):
    return NotImplementedError


class OneHot(Cell):

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
        super(OneHot, self).__init__()
        self.onehot = P.OneHot(axis)
        self.depth = depth
        self.dtype = dtype
        self.on_value = F.cast(on_value, self.dtype)
        self.off_value = F.cast(off_value, self.dtype)

    def construct(self, indices):
        return self.onehot(indices, self.depth, self.on_value, self.off_value)


class L2Normalize(Cell):

    def __init__(self, axis=None, epsilon=1e-12):
        super(L2Normalize, self).__init__()
        pass

    def construct(self, input, *args, **kwargs):
        pass


class EmbeddingLookup(Cell):

    def __init__(self, max_norm=0):
        super(EmbeddingLookup, self).__init__()
        self.max_norm = max_norm
        self.embedding_lookup = P.EmbeddingLookup()

    def construct(self, params, ids, *args, **kwargs):
        return self.embedding_lookup(params, ids, self.max_norm)


class NCELoss(Cell):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        super(NCELoss, self).__init__()
        pass

    def construct(self, weights, biases, labels, inputs, num_sampled, num_classes):
        raise NotImplementedError


class NotEqual(Cell):

    def __init__(self):
        super(NotEqual, self).__init__()
        self.not_equal = P.NotEqual()

    def construct(self, x, y):
        outputs = self.not_equal(x, y)
        return outputs


class CountNonzero(object):

    def __init__(self, keepdims=None, dtype=int64):
        self.keepdims = keepdims
        self.dtype = dtype

    def __call__(self, input, axis=None):
        input = self.convert_dtype(input)
        return count_nonzero(x=input, axis=axis, keep_dims=self.keepdims, dtype=self.dtype)

    def bool_convert_to_tensor(self, x):
        x = x.asnumpy()
        shapes = x.shape
        b = np.ones(shapes)
        if len(shapes) == 1:
            for i in range(shapes - 1):
                if x[i] ==True:
                    b[i] = 1
                else:
                    b[i] = 0
        if len(shapes) == 2:
            for i in range(shapes[0] - 1):
                for j in range(shapes[1] - 1):
                    if x[i][j] ==True:
                        b[i][j] = 1
                    else:
                        b[i][j] = 0
        return Tensor(b, dtype=float32)

    def convert_dtype(self, input):
        if input.shape == 1 and type(input[0]) is bool:
            output = self.bool_convert_to_tensor(input)
        elif input.shape == 2 and type(input[0][0]) is bool:
            output = self.bool_convert_to_tensor(input)
        else:
            output = input
        return output


class Resize(Cell):

    def __init__(self, scale, method, antialias=False, data_format='channels_last', ksize=None):
        super(Resize, self).__init__()
        self.data_format = data_format
        if method not in ['nearest', 'bilinear']:
            raise ('The method must be "nearest" or "bilinear".')
        self.method = method

        if ksize is None:
            raise ('The "bilinear" and  "nearest" method must enter ksize. The dimension of size must be 2 (H, W).')

        out_seize = (int(ksize[0] * scale[0]), int(ksize[1] * scale[1]))
        if self.method == 'nearest':
            self.resize = P.ResizeNearestNeighbor(size=out_seize, align_corners=antialias)
        elif self.method == 'bilinear':

            self.resize = P.ResizeBilinear(size=out_seize)

    def construct(self, inputs):
        if self.data_format == 'channels_last':
            inputs = nhwc_to_nchw(inputs)
        outputs = self.resize(inputs)
        if self.data_format == 'channels_last':
            outputs = nchw_to_nhwc(outputs)
        return outputs


def resize(inputs, output_size, method, antialias):
    raise NotImplementedError


class ZeroPadding1D(Cell):

    def __init__(self, padding):
        super(ZeroPadding1D, self).__init__()
        if np.size(padding) == 2:
            self.pad = P.Pad(paddings=padding)
        else:
            raise ("The shape of parameter paddings is (N, 2). N is the rank of input data.")

    def construct(self, inputs):
        return self.pad(inputs)


class ZeroPadding2D(Cell):

    def __init__(self, padding):
        super(ZeroPadding2D, self).__init__()
        if np.size(padding) == 4:
            self.pad = P.Pad(paddings=padding)
        else:
            raise ("The shape of parameter paddings is (N, 2). N is the rank of input data.")

    def construct(self, inputs):
        return self.pad(inputs)


class ZeroPadding3D(Cell):

    def __init__(self, padding):
        super(ZeroPadding3D, self).__init__()
        if np.size(padding) == 6:
            self.pad = P.Pad(paddings=padding)
        else:
            raise ("The shape of parameter paddings is (N, 2). N is the rank of input data.")

    def construct(self, inputs):
        return self.pad(inputs)


class Sign(Cell):

    def __init__(self):
        super(Sign, self).__init__()
        self.sign = P.Sign()

    def construct(self, x):
        return self.sign(x)


class Ceil(Cell):

    def __init__(self):
        super(Ceil, self).__init__()
        self.ceil = P.Ceil()

    def construct(self, x):
        return self.ceil(x)


def ceil(x):
    _ceil = P.Ceil()
    return _ceil(x)


def multiply(x, y):
    raise NotImplementedError


def divide(x, y):
    return msnp.divide(x, y)


def identity(x):
    raise NotImplementedError


class BatchToSpace(Cell):

    def __init__(self, block_size, crops):
        super(BatchToSpace, self).__init__()
        self.batch_to_space = P.BatchToSpace(block_size=block_size, crops=crops)

    def __call__(self, input_x):
        return self.batch_to_space(input_x)


class DepthToSpace(Cell):

    def __init__(self, block_size, data_format='NHWC'):
        super(DepthToSpace, self).__init__()
        self.data_format = data_format
        self.depth_to_space = P.DepthToSpace(block_size=block_size)

    def __call__(self, input):
        if self.data_format == 'NHWC':
            input = nhwc_to_nchw(input)

        output = self.depth_to_space(input)

        if self.data_format == 'NHWC':
            output = nchw_to_nhwc(output)

        return output
