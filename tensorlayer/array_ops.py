#! /usr/bin/python
# -*- coding: utf-8 -*-
"""A file containing functions related to array manipulation."""

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops.array_ops import shape_internal
from tensorflow.python.ops.gen_array_ops import fill
from tensorflow.python.ops.gen_array_ops import reshape

__all__ = ['alphas', 'alphas_like']


def alphas(shape, alpha_value, name=None):
    """Creates a tensor with all elements set to `alpha_value`.
    This operation returns a tensor of type `dtype` with shape `shape` and all
    elements set to alpha.

    Parameters
    ----------
    shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type `int32`.
        The shape of the desired tensor
    alpha_value: `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, int32`, `int64`
        The value used to fill the resulting `Tensor`.
    name: str
        A name for the operation (optional).

    Returns
    -------
    A `Tensor` with all elements set to alpha.

    Examples
    --------
    >>> tl.alphas([2, 3], tf.int32)  # [[alpha, alpha, alpha], [alpha, alpha, alpha]]
    """

    with ops.name_scope(name, "alphas", [shape]) as name:

        alpha_tensor = convert_to_tensor(alpha_value)
        alpha_dtype = dtypes.as_dtype(alpha_tensor.dtype).base_dtype

        if not isinstance(shape, ops.Tensor):
            try:
                shape = constant_op._tensor_shape_tensor_conversion_function(tensor_shape.TensorShape(shape))
            except (TypeError, ValueError):
                shape = ops.convert_to_tensor(shape, dtype=dtypes.int32)

        if not shape._shape_tuple():
            shape = reshape(shape, [-1])  # Ensure it's a vector

        try:
            output = constant(alpha_value, shape=shape, dtype=alpha_dtype, name=name)

        except (TypeError, ValueError):
            output = fill(shape, constant(alpha_value, dtype=alpha_dtype), name=name)

        if output.dtype.base_dtype != alpha_dtype:
            raise AssertionError("Dtypes do not corresponds: %s and %s" % (output.dtype.base_dtype, alpha_dtype))

        return output


def alphas_like(tensor, alpha_value, name=None, optimize=True):
    """Creates a tensor with all elements set to `alpha_value`.
    Given a single tensor (`tensor`), this operation returns a tensor of the same
    type and shape as `tensor` with all elements set to `alpha_value`.

    Parameters
    ----------
    tensor: tf.Tensor
        The Tensorflow Tensor that will be used as a template.
    alpha_value: `float32`, `float64`, `int8`, `uint8`, `int16`, `uint16`, int32`, `int64`
        The value used to fill the resulting `Tensor`.
    name: str
        A name for the operation (optional).
    optimize: bool
        if true, attempt to statically determine the shape of 'tensor' and encode it as a constant.

    Returns
    -------
    A `Tensor` with all elements set to `alpha_value`.

    Examples
    --------
    >>> tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    >>> tl.alphas_like(tensor, 0.5)  # [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    """

    with ops.name_scope(name, "alphas_like", [tensor]) as name:
        tensor = ops.convert_to_tensor(tensor, name="tensor")

        if context.in_eager_mode():  #and dtype is not None and dtype != tensor.dtype:
            ret = alphas(shape_internal(tensor, optimize=optimize), alpha_value=alpha_value, name=name)

        else:  # if context.in_graph_mode():

            # For now, variant types must be created via zeros_like; as we need to
            # pass the input variant object to the proper zeros callback.

            if (optimize and tensor.shape.is_fully_defined()):
                # We can produce a zeros tensor independent of the value of 'tensor',
                # since the shape is known statically.
                ret = alphas(tensor.shape, alpha_value=alpha_value, name=name)

            # elif dtype is not None and dtype != tensor.dtype and dtype != dtypes.variant:
            else:
                ret = alphas(shape_internal(tensor, optimize=optimize), alpha_value=alpha_value, name=name)

            ret.set_shape(tensor.get_shape())

        return ret
