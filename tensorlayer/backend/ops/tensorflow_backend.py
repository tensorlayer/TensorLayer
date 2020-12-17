#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from .tensorflow_nn import nchw_to_nhwc, nhwc_to_nchw
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

DType = tf.DType
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64
int8 = tf.int8
int16 = tf.int16
int32 = tf.int32
int64 = tf.int64
uint8 = tf.uint8
uint16 = tf.uint16
uint32 = tf.uint32
uint64 = tf.uint64

# isinstance input output
# TensorLike = tf_ops._TensorLike


def set_context(**kwargs):
    raise Exception("Using TenosrFlow backend,You don't need to set context")


def get_tensor_shape(x):
    return x.get_shape().as_list()


# initializers
def zeros(shape, dtype=tf.float32):
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
    return tf.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype=tf.float32):
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
    return tf.ones(shape=shape, dtype=dtype)


def constant(value, dtype=tf.float32, shape=None):
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
    return tf.constant(value=value, dtype=dtype, shape=shape)


def random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None):
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
    outputs = tf.random.uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)
    return outputs


def random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None):
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
    outputs = tf.random.normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    return outputs


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None):
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
    outputs = tf.random.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)
    return outputs


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
    return tf.initializers.he_normal(seed)(shape=shape, dtype=dtype)


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

    var = tf.Variable(initial_value=initial_value, name=name, trainable=trainable)
    return var


class MatMul(object):

    def __init__(self):
        pass

    def __call__(self, a, b):
        return tf.matmul(a, b)


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

    outputs = tf.matmul(a, b)
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

    Returns
    -------
        A Tensor. Has the same type as a.
    """

    outputs = tf.add(value, bias)
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


def minimum(x, y):
    """
    Returns the min of x and y (i.e. x < y ? x : y) element-wise.

    Parameters
    ----------
    x : tensor.
        Must be one of the following types: bfloat16, half, float32, float64, int32, int64.
    y : A Tensor.
        Must have the same type as x.

    Returns
    -------
        A Tensor. Has the same type as x
    """

    outputs = tf.minimum(x=x, y=y)
    return outputs


class FlattenReshape(object):

    def __init__(self):
        pass

    def __call__(self, inputs):
        dim = 1
        for d in get_tensor_shape(inputs)[1:]:
            dim *= d
        return tf.reshape(inputs, [-1, dim])


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def __call__(self, tensor):
        return tf.reshape(tensor, self.shape)


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

    return tf.reshape(tensor, shape)


class Concat(object):

    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __call__(self, values):
        return tf.concat(values=values, axis=self.axis)


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

    return tf.concat(values, axis)


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

    return tf.convert_to_tensor(value, dtype)


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
    return tf.sqrt(x)


class ReduceSum(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, input):
        return tf.reduce_sum(input, axis=self.axis)


class ReduceMean(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, inputs):
        output = tf.reduce_mean(inputs, self.axis)
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

    return tf.reduce_mean(input_tensor, axis=axis)


class ReduceMax(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, inputs):
        output = tf.reduce_max(inputs, self.axis)
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

    return tf.reduce_max(input_tensor, axis=axis)


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

    return tf.reduce_min(input_tensor, axis=axis)


def pad(tensor, paddings, mode='CONSTANT', constant_values=0):
    """
    Pads a tensor.

    Parameters
    ----------
    tensor : tensor
        A Tensor.
    paddings : tensor
        A Tensor of type int32.
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
    outputs = tf.pad(tensor, paddings, mode=mode, constant_values=constant_values)
    return outputs


class Unstack(object):

    def __init__(self, axis, num=None):
        self.axis = axis
        self.num = num

    def __call__(self, values):
        return tf.unstack(values, num=self.num, axis=self.axis)


class Stack(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, values):
        return tf.stack(values, axis=self.axis)


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

    return tf.stack(values, axis=axis)


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

    return tf.meshgrid(x, y)


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

    if limit is None:
        outputs = tf.range(start, delta=delta, dtype=dtype)
    else:
        outputs = tf.range(start, limit, delta=delta, dtype=dtype)
    return outputs


class ExpandDims(object):

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, input):
        return tf.expand_dims(input, axis=self.axis)


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

    return tf.expand_dims(input, axis)


class Tile(object):

    def __init__(self):
        pass

    def __call__(self, input, multiples):
        return tf.tile(input, multiples)


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

    return tf.tile(input, multiples)


class Cast(object):

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        return tf.cast(x, dtype=self.dtype)


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

    return tf.cast(x, dtype=dtype)


class Transpose(object):

    def __init__(self, perm, conjugate=False):
        self.perm = perm
        self.conjugate = conjugate

    def __call__(self, a):
        return tf.transpose(a, self.perm, self.conjugate)


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
        Setting it to True is mathematically equivalent to tf.math.conj(tf.transpose(input)).

    Returns
    -------
        A transposed Tensor.
    """

    return tf.transpose(a, perm, conjugate)


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

    return tf.gather_nd(params, indices, batch_dims)


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

    return tf.clip_by_value(t, clip_value_min, clip_value_max)


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

    return tf.split(value=value, num_or_size_splits=num_or_size_splits, axis=axis, num=num)


def floor(x):
    return tf.floor(x)


def gather(params, indices):
    return tf.gather(params, indices)


def linspace(start, stop, num):
    return tf.linspace(start, stop, num)


def slice(inputs, starts, sizes):
    return tf.slice(inputs, starts, sizes)


def add_n(inputs):
    return tf.add_n(inputs)


class OneHot(object):

    def __init__(self, depth, on_value, off_value, axis, dtype):
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype

    def __call__(self, inputs, *args, **kwargs):
        outputs = tf.one_hot(
            inputs, self.depth, on_value=self.on_value, off_value=self.off_value, axis=self.axis, dtype=self.dtype
        )
        return outputs


class L2Normalize(object):

    def __init__(self, axis=None, epsilon=1e-12):
        self.axis = axis
        self.epsilon = epsilon

    def __call__(self, input, *args, **kwargs):
        outputs = tf.math.l2_normalize(input, axis=self.axis, epsilon=self.epsilon)
        return outputs


class EmbeddingLookup(object):

    def __init__(self, max_norm=None):
        self.max_norm = max_norm

    def __call__(self, params, ids, *args, **kwargs):
        outputs = tf.nn.embedding_lookup(params=params, ids=ids, max_norm=self.max_norm)
        return outputs


class NCELoss(object):

    def __init__(self, num_true=1, sampled_values=None, remove_accidental_hits=False):
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits

    def __call__(self, weights, biases, labels, inputs, num_sampled, num_classes):
        outputs = tf.nn.nce_loss(
            weights=weights, biases=biases, inputs=inputs, labels=labels, num_sampled=num_sampled,
            num_classes=num_classes
        )
        return outputs


class Not_equal(object):

    def __init__(self):
        pass

    def __call__(self, x, y):
        return tf.not_equal(x, y)


class Count_nonzero(object):

    def __init__(self, keepdims=None, dtype=int64):
        self.keepdims = keepdims
        self.dtype = dtype

    def __call__(self, input, axis=None):
        return tf.math.count_nonzero(input, axis=axis, keepdims=self.keepdims, dtype=self.dtype)



class Resize:

    def __init__(self, scale, method, antialias=False, data_format='channels_last', ksize=None):
        self.method = method
        self.antialias = antialias
        self.scale = scale
        self.data_format = data_format

    def __call__(self, inputs):
        if self.data_format == 'channels_first':
            inputs = nchw_to_nhwc(inputs)
        if len(get_tensor_shape(inputs)) == 4:
            output_size = [int(inputs.shape[1] * self.scale[0]), int(inputs.shape[2] * self.scale[1])]
        else:
            raise ("The inputs shape must be 4-D Tensor.")
        outputs = tf.image.resize(inputs, size=output_size, method=self.method, antialias=self.antialias)
        if self.data_format == 'channels_first':
            outputs = nhwc_to_nchw(outputs)
        return outputs


def resize(inputs, output_size, method, antialias):
    return tf.image.resize(inputs, size=output_size, method=method, antialias=antialias)


class ZeroPadding1D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding1D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class ZeroPadding2D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding2D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class ZeroPadding3D(object):

    def __init__(self, padding):
        self.zeropad = tf.keras.layers.ZeroPadding3D(padding=padding)

    def __call__(self, inputs):
        return self.zeropad(inputs)


class Sign(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return tf.sign(x)
