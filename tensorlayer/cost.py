import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

## Cost Functions
def cross_entropy(output, target):
    """Return the cost function of cross-entropy between two distribution.

    Parameters
    ----------
    output : tensorflow variable
        A distribution with shape: [None, n_feature].
    target : tensorflow variable
        A distribution with shape: [None, n_feature].

    Examples
    --------
    >>> xxx
    >>> xxx

    Notes
    -----
    About cross-entropy:
    https://en.wikipedia.org/wiki/Cross_entropy
    The code are borrowed from:
    https://github.com/cmgreen210/TensorFlowDeepAutoencoder/blob/master/code/ae/autoencoder.py
    """
    with tf.name_scope("cross_entropy_loss"):
        net_output_tf = output
        target_tf = target
        cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name=None),target_tf),
                             tf.mul(tf.log(1 - net_output_tf), (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='cross_entropy_mean')

## Regularization Functions
def li_regularizer(scale):
  """Returns a function that can be used to apply group li regularization to weights.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  li regularization removes the neurons of previous layer.
  'i' represents 'inputs'

  Parameters
  ----------
  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

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
    #   return standard_ops.mul(
    #       my_scale,
    #       standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(weights**2, 1))),
    #       name=scope)
    return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
        #   standard_ops.reduce_mean(standard_ops.sqrt(standard_ops.reduce_mean(tf.square(weights), 1))),
          name=scope)
  return li

def lo_regularizer(scale):
  """Returns a function that can be used to apply group lo regularization to weights.
  see:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  Lo regularization removes the neurons of current layer.
  'o' represents outputs

  Parameters
  ----------
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

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
    #   return standard_ops.mul(
    #       my_scale,
    #       standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(weights**2, 0))),
    #       name=scope)
      return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 0))),
        #   standard_ops.reduce_mean(standard_ops.sqrt(standard_ops.reduce_mean(tf.square(weights), 0))),
          name=scope)
  return lo

def maxnorm_regularizer(scale=1.0):
  """Returns a function that can be used
  to apply max-norm regularization to weights.
  see:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
  Max-norm regularization

  Parameters
  ----------
  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

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
  """Returns a function that can be
  used to apply max-norm regularization to each column of weight matrix.
  see:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
  Max-norm output regularization removes the neurons of current layer.

  Parameters
  ----------
  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

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
  """Returns a function that can be
  used to apply max-norm regularization to each row of weight matrix.
  see:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
  Max-norm output regularization removes the neurons of current layer.

  Parameters
  ----------
  scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

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
