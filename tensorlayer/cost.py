#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

## Cost Functions
def cross_entropy(output, target, name="cross_entropy_loss"):
    """Returns the TensorFlow expression of cross-entropy of two distributions, implement
    softmax internally.

    Parameters
    ----------
    output : Tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    target : Tensorflow variable
        A distribution with shape: [batch_size, n_feature].

    Examples
    --------
    >>> ce = tf.cost.cross_entropy(y_logits, y_target_logits)

    References
    -----------
    - About cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    - The code is borrowed from: `here <https://en.wikipedia.org/wiki/Cross_entropy>`_.
    """
    with tf.name_scope(name):
        # net_output_tf = output
        # target_tf = target
        # cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name=None),target_tf),
        #                      tf.mul(tf.log(1 - net_output_tf), (1 - target_tf)))
        # return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='cross_entropy_mean')
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, target))

# Undocumented
def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    preds : A `Tensor` of type `float32` or `float64`.
    targets : A `Tensor` of the same type and shape as `preds`.
    """
    print("Undocumented")
    from tensorflow.python.framework import ops
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))


def mean_squared_error(output, target):
    """Return the TensorFlow expression of mean-squre-error of two distributions.

    Parameters
    ----------
    output : tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    target : tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    """
    with tf.name_scope("mean_squared_error_loss"):
        mse = tf.reduce_sum(tf.squared_difference(output, target), reduction_indices = 1)
        return tf.reduce_mean(mse)

def cross_entropy_seq(logits, target_seqs, batch_size=1, num_steps=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for Fixed Length RNN outputs.

    Parameters
    ----------
    logits : Tensorflow variable
        2D tensor, ``network.outputs``, [batch_size*n_steps (n_examples), number of output units]
    target_seqs : Tensorflow variable
        target : 2D tensor [batch_size, n_steps], if the number of step is dynamic, please use ``cross_entropy_seq_with_mask`` instead.
    batch_size : a int, default is 1
        RNN batch_size, number of concurrent processes, divide the loss by batch_size.
    num_steps : a int
        sequence length

    Examples
    --------
    >>> see PTB tutorial for more details
    >>> input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> cost = tf.cost.cross_entropy_seq(network.outputs, targets, batch_size, num_steps)
    """
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(target_seqs, [-1])],
        [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss) / batch_size
    return cost


def cross_entropy_seq_with_mask(logits, target_seqs, input_mask, return_details=False):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for Dynamic RNN outputs.

    Parameters
    -----------
    logits : network identity outputs
        2D tensor, ``network.outputs``, [batch_size, number of output units].
    target_seqs : int of tensor, like word ID.
        [batch_size, ?]
    input_mask : the mask to compute loss
        The same size with target_seqs, normally 0 and 1.
    return_details : boolean
        If False (default), only returns the loss

        If True, returns the loss, losses, weights and targets (reshape to one vetcor)

    Examples
    --------
    - see Image Captioning Example.
    """
    print("     cross_entropy_seq_with_mask : Undocumented")
    targets = tf.reshape(target_seqs, [-1])   # to one vector
    weights = tf.to_float(tf.reshape(input_mask, [-1]))   # to one vector like targets
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
    loss = tf.div(tf.reduce_sum(tf.mul(losses, weights)),   # loss from mask. reduce_sum before element-wise mul with mask !!
                    tf.reduce_sum(weights),
                    name="seq_loss_with_mask")
    if return_details:
        return loss, losses, weights, targets
    else:
        return loss

## Regularization Functions
def li_regularizer(scale):
  """li regularization removes the neurons of previous layer, `i` represents `inputs`.\n
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
  ValueError : if scale is outside of the range [0.0, 1.0] or if scale is not a float.
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
  """lo regularization removes the neurons of current layer, `o` represents `outputs`\n
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
  ValueError : If scale is outside of the range [0.0, 1.0] or if scale is not a float.
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
  ValueError : If scale is outside of the range [0.0, 1.0] or if scale is not a float.
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
  ValueError : If scale is outside of the range [0.0, 1.0] or if scale is not a float.
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
  ValueError : If scale is outside of the range [0.0, 1.0] or if scale is not a float.
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
