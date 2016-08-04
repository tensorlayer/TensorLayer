#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

## Cost Functions
def cross_entropy(output, target):
    """Returns the cost function of Cross-entropy of two distributions, implement
    softmax internally.

    Parameters
    ----------
    output : Tensorflow variable
        A distribution with shape: [None, n_feature].
    target : Tensorflow variable
        A distribution with shape: [None, n_feature].

    Examples
    --------
    >>> ce = tf.cost.cross_entropy(y_logits, y_target_logits)

    Notes
    -----
    About cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    The code is borrowed from: `here <https://en.wikipedia.org/wiki/Cross_entropy>`_.
    """
    with tf.name_scope("cross_entropy_loss"):
        net_output_tf = output
        target_tf = target
        cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name=None),target_tf),
                             tf.mul(tf.log(1 - net_output_tf), (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='cross_entropy_mean')

def mean_squared_error(output, target):
    """Return the cost function of Mean-squre-error of two distributions.

    Parameters
    ----------
    output : tensorflow variable
        A distribution with shape: [None, n_feature].
    target : tensorflow variable
        A distribution with shape: [None, n_feature].

    """
    mse = tf.reduce_sum(tf.squared_difference(y, x_recon), reduction_indices = 1)
    return tf.reduce_mean(mse)

## Regularization Functions
def li_regularizer(scale):
  """li regularization removes the neurons of previous layer, 'i' represents 'inputs'.\n
  Returns a function that can be used to apply group li regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.



  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  --------
  A function with signature `li(weights, name=None)` that apply L1 regularization.

  Raises
  ------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops
  # from tensorflow.python.platform import tf_logging as logging

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def li(weights, name=None):
    """Applies li regularization to weights."""
    with ops.op_scope([weights], name, 'li_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
    return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
          name=scope)
  return li

def lo_regularizer(scale):
  """lo regularization removes the neurons of current layer, 'o' represents outputs\n
  Returns a function that can be used to apply group lo regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  -------
  A function with signature `lo(weights, name=None)` that apply Lo regularization.

  Raises
  ------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops
  # from tensorflow.python.platform import tf_logging as logging

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def lo(weights, name=None):
    """Applies group column regularization to weights."""
    with ops.op_scope([weights], name, 'lo_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 0))),
          name=scope)
  return lo

def maxnorm_regularizer(scale=1.0):
  """Max-norm regularization returns a function that can be used
  to apply max-norm regularization to weights.
  About max-norm: `wiki <https://en.wikipedia.org/wiki/Matrix_norm#Max_norm>`_.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn(weights, name=None)` that apply Lo regularization.

  Raises
  --------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    # if scale >= 1.:
    #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
    #                    scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def mn(weights, name=None):
    """Applies max-norm regularization to weights."""
    with ops.op_scope([weights], name, 'maxnorm_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(my_scale, standard_ops.reduce_max(standard_ops.abs(weights)), name=scope)
  return mn

def maxnorm_o_regularizer(scale):
  """Max-norm output regularization removes the neurons of current layer.\n
  Returns a function that can be used to apply max-norm regularization to each column of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn_o(weights, name=None)` that apply Lo regularization.

  Raises
  ---------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    # if scale >= 1.:
    #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
    #                    scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def mn_o(weights, name=None):
    """Applies max-norm regularization to weights."""
    with ops.op_scope([weights], name, 'maxnorm_o_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                               name='scale')
      return standard_ops.mul(my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 0)), name=scope)
  return mn_o

def maxnorm_i_regularizer(scale):
  """Max-norm input regularization removes the neurons of previous layer.\n
  Returns a function that can be used to apply max-norm regularization to each row of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns
  ---------
  A function with signature `mn_i(weights, name=None)` that apply Lo regularization.

  Raises
  ---------
  ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    # if scale >= 1.:
    #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
    #                    scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def mn_i(weights, name=None):
    """Applies max-norm regularization to weights."""
    with ops.op_scope([weights], name, 'maxnorm_o_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                               name='scale')
      return standard_ops.mul(my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 1)), name=scope)
  return mn_i





#
