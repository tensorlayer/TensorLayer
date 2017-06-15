#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

## Cost Functions

def cross_entropy(output, target, name=None):
    """It is a softmax cross-entropy operation, returns the TensorFlow expression of cross-entropy of two distributions, implement
    softmax internally. See ``tf.nn.sparse_softmax_cross_entropy_with_logits``.

    Parameters
    ----------
    output : Tensorflow variable
        A distribution with shape: [batch_size, n_feature].
    target : Tensorflow variable
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.

    Examples
    --------
    >>> ce = tl.cost.cross_entropy(y_logits, y_target_logits, 'my_loss')

    References
    -----------
    - About cross-entropy: `wiki <https://en.wikipedia.org/wiki/Cross_entropy>`_.\n
    - The code is borrowed from: `here <https://en.wikipedia.org/wiki/Cross_entropy>`_.
    """
    try: # old
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, targets=target))
    except: # TF 1.0
        assert name is not None, "Please give a unique name to tl.cost.cross_entropy for TF1.0+"
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name=name))

def sigmoid_cross_entropy(output, target, name=None):
    """It is a sigmoid cross-entropy operation, see ``tf.nn.sigmoid_cross_entropy_with_logits``.
    """
    try: # TF 1.0
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output, name=name))
    except:
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, targets=target))


def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    """Computes binary cross entropy given `output`.

    For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Parameters
    ----------
    output : tensor of type `float32` or `float64`.
    target : tensor of the same type and shape as `output`.
    epsilon : float
        A small value to avoid output is zero.
    name : string
        An optional name to attach to this layer.

    References
    -----------
    - `DRAW <https://github.com/ericjang/draw/blob/master/draw.py#L73>`_
    """
#     from tensorflow.python.framework import ops
#     with ops.op_scope([output, target], name, "bce_loss") as name:
#         output = ops.convert_to_tensor(output, name="preds")
#         target = ops.convert_to_tensor(targets, name="target")
    with tf.name_scope(name):
        return tf.reduce_mean(tf.reduce_sum(-(target * tf.log(output + epsilon) +
                              (1. - target) * tf.log(1. - output + epsilon)), axis=1))


def mean_squared_error(output, target, is_mean=False):
    """Return the TensorFlow expression of mean-squre-error of two distributions.

    Parameters
    ----------
    output : 2D or 4D tensor.
    target : 2D or 4D tensor.
    is_mean : boolean, if True, use ``tf.reduce_mean`` to compute the loss of one data, otherwise, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:   # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 4: # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
        return mse

def normalized_mean_square_error(output, target):
    """Return the TensorFlow expression of normalized mean-squre-error of two distributions.

    Parameters
    ----------
    output : 2D or 4D tensor.
    target : 2D or 4D tensor.
    """
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:   # [batch_size, n_feature]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=1))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=1))
        elif output.get_shape().ndims == 4: # [batch_size, w, h, c]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1,2,3]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1,2,3]))
        nmse = tf.reduce_mean(nmse_a / nmse_b)
    return nmse


def dice_coe(output, target, epsilon=1e-10):
    """Sørensen–Dice coefficient for comparing the similarity of two distributions,
    usually be used for binary image segmentation i.e. labels are binary.
    The coefficient = [0, 1], 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    epsilon : float
        An optional name to attach to this layer.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_, epsilon=1e-5)

    References
    -----------
    - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    # inse = tf.reduce_sum( tf.mul(output, target) )
    # l = tf.reduce_sum( tf.mul(output, output) )
    # r = tf.reduce_sum( tf.mul(target, target) )
    inse = tf.reduce_sum( output * target )
    l = tf.reduce_sum( output * output )
    r = tf.reduce_sum( target * target )
    dice = 2 * (inse) / (l + r)
    if epsilon == 0:
        return dice
    else:
        return tf.clip_by_value(dice, 0, 1.0-epsilon)


def dice_hard_coe(output, target, epsilon=1e-10):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity of two distributions,
    usually be used for binary image segmentation i.e. labels are binary.
    The coefficient = [0, 1], 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    epsilon : float
        An optional name to attach to this layer.

    Examples
    ---------
    >>> outputs = pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - dice_coe(outputs, y_, epsilon=1e-5)

    References
    -----------
    - `wiki-dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    output = tf.cast(output > 0.5, dtype=tf.float32)
    target = tf.cast(target > 0.5, dtype=tf.float32)
    inse = tf.reduce_sum( output * target )
    l = tf.reduce_sum( output * output )
    r = tf.reduce_sum( target * target )
    dice = 2 * (inse) / (l + r)
    if epsilon == 0:
        return dice
    else:
        return tf.clip_by_value(dice, 0, 1.0-epsilon)

def iou_coe(output, target, threshold=0.5, epsilon=1e-10):
    """Non-differentiable Intersection over Union, usually be used for evaluating binary image segmentation.
    The coefficient = [0, 1], 1 means totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    epsilon : float
        A small value to avoid zero denominator when both output and target output nothing.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> iou = tl.cost.iou_coe(outputs[:,:,:,0], y_[:,:,:,0])

    Notes
    ------
    - IOU cannot be used as training loss, people usually use dice coefficient for training, and IOU for evaluating.
    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    intersection = tf.reduce_sum(pre * truth)
    union = tf.reduce_sum(tf.cast((pre + truth) > threshold, dtype=tf.float32))
    return tf.reduce_sum(intersection) / (tf.reduce_sum(union) + epsilon)


def cross_entropy_seq(logits, target_seqs, batch_size=None):#, batch_size=1, num_steps=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for Fixed Length RNN outputs.

    Parameters
    ----------
    logits : Tensorflow variable
        2D tensor, ``network.outputs``, [batch_size*n_steps (n_examples), number of output units]
    target_seqs : Tensorflow variable
        target : 2D tensor [batch_size, n_steps], if the number of step is dynamic, please use ``cross_entropy_seq_with_mask`` instead.
    batch_size : None or int.
        If not None, the return cost will be divided by batch_size.

    Examples
    --------
    >>> see PTB tutorial for more details
    >>> input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> cost = tl.cost.cross_entropy_seq(network.outputs, targets)
    """
    try: # TF 1.0
        sequence_loss_by_example_fn = tf.contrib.legacy_seq2seq.sequence_loss_by_example
    except:
        sequence_loss_by_example_fn = tf.nn.seq2seq.sequence_loss_by_example

    loss = sequence_loss_by_example_fn(
        [logits],
        [tf.reshape(target_seqs, [-1])],
        [tf.ones_like(tf.reshape(target_seqs, [-1]), dtype=tf.float32)])
        # [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss) #/ batch_size
    if batch_size is not None:
        cost = cost / batch_size
    return cost


def cross_entropy_seq_with_mask(logits, target_seqs, input_mask, return_details=False, name=None):
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
        - If False (default), only returns the loss.
        - If True, returns the loss, losses, weights and targets (reshape to one vetcor).

    Examples
    --------
    - see Image Captioning Example.
    """
    targets = tf.reshape(target_seqs, [-1])   # to one vector
    weights = tf.to_float(tf.reshape(input_mask, [-1]))   # to one vector like targets
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name) * weights
    #losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name)) # for TF1.0 and others

    try: ## TF1.0
        loss = tf.divide(tf.reduce_sum(losses),   # loss from mask. reduce_sum before element-wise mul with mask !!
                        tf.reduce_sum(weights),
                        name="seq_loss_with_mask")
    except: ## TF0.12
        loss = tf.div(tf.reduce_sum(losses),   # loss from mask. reduce_sum before element-wise mul with mask !!
                        tf.reduce_sum(weights),
                        name="seq_loss_with_mask")
    if return_details:
        return loss, losses, weights, targets
    else:
        return loss


def cosine_similarity(v1, v2):
    """Cosine similarity [-1, 1], `wiki <https://en.wikipedia.org/wiki/Cosine_similarity>`_.

    Parameters
    -----------
    v1, v2 : tensor of [batch_size, n_feature], with the same number of features.

    Returns
    -----------
    a tensor of [batch_size, ]
    """
    try: ## TF1.0
        cost = tf.reduce_sum(tf.multiply(v1, v2), 1) / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))
    except: ## TF0.12
        cost = tf.reduce_sum(tf.mul(v1, v2), reduction_indices=1) / (tf.sqrt(tf.reduce_sum(tf.mul(v1, v1), reduction_indices=1)) * tf.sqrt(tf.reduce_sum(tf.mul(v2, v2), reduction_indices=1)))
    return cost


## Regularization Functions
def li_regularizer(scale, scope=None):
  """li regularization removes the neurons of previous layer, `i` represents `inputs`.\n
  Returns a function that can be used to apply group li regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  scope: An optional scope name for TF12+.

  Returns
  --------
  A function with signature `li(weights, name=None)` that apply Li regularization.

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
    with tf.name_scope('li_regularizer') as scope:
        my_scale = ops.convert_to_tensor(scale,
                                           dtype=weights.dtype.base_dtype,
                                           name='scale')
        if tf.__version__ <= '0.12':
            standard_ops_fn = standard_ops.mul
        else:
            standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
              my_scale,
              standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
              name=scope)
  return li



def lo_regularizer(scale, scope=None):
  """lo regularization removes the neurons of current layer, `o` represents `outputs`\n
  Returns a function that can be used to apply group lo regularization to weights.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  scope: An optional scope name for TF12+.

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

  def lo(weights, name='lo_regularizer'):
    """Applies group column regularization to weights."""
    with tf.name_scope(name) as scope:
        my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
        if tf.__version__ <= '0.12':
            standard_ops_fn = standard_ops.mul
        else:
            standard_ops_fn = standard_ops.multiply
        return standard_ops_fn(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 0))),
          name=scope)
  return lo

def maxnorm_regularizer(scale=1.0, scope=None):
  """Max-norm regularization returns a function that can be used
  to apply max-norm regularization to weights.
  About max-norm: `wiki <https://en.wikipedia.org/wiki/Matrix_norm#Max_norm>`_.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  scope: An optional scope name.

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

  def mn(weights, name='max_regularizer'):
    """Applies max-norm regularization to weights."""
    with tf.name_scope(name) as scope:
          my_scale = ops.convert_to_tensor(scale,
                                           dtype=weights.dtype.base_dtype,
                                           name='scale')
          if tf.__version__ <= '0.12':
              standard_ops_fn = standard_ops.mul
          else:
              standard_ops_fn = standard_ops.multiply
          return standard_ops_fn(my_scale, standard_ops.reduce_max(standard_ops.abs(weights)), name=scope)
  return mn

def maxnorm_o_regularizer(scale, scope):
  """Max-norm output regularization removes the neurons of current layer.\n
  Returns a function that can be used to apply max-norm regularization to each column of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  scope: An optional scope name.

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

  def mn_o(weights, name='maxnorm_o_regularizer'):
     """Applies max-norm regularization to weights."""
     with tf.name_scope(name) as scope:
          my_scale = ops.convert_to_tensor(scale,
                                           dtype=weights.dtype.base_dtype,
                                                   name='scale')
          if tf.__version__ <= '0.12':
             standard_ops_fn = standard_ops.mul
          else:
             standard_ops_fn = standard_ops.multiply
          return standard_ops_fn(my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 0)), name=scope)
  return mn_o

def maxnorm_i_regularizer(scale, scope=None):
  """Max-norm input regularization removes the neurons of previous layer.\n
  Returns a function that can be used to apply max-norm regularization to each row of weight matrix.\n
  The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`_.

  Parameters
  ----------
  scale : float
    A scalar multiplier `Tensor`. 0.0 disables the regularizer.
  scope: An optional scope name.

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

  def mn_i(weights, name='maxnorm_i_regularizer'):
     """Applies max-norm regularization to weights."""
     with tf.name_scope(name) as scope:
          my_scale = ops.convert_to_tensor(scale,
                                           dtype=weights.dtype.base_dtype,
                                                   name='scale')
          if tf.__version__ <= '0.12':
             standard_ops_fn = standard_ops.mul
          else:
             standard_ops_fn = standard_ops.multiply
          return standard_ops_fn(my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 1)), name=scope)
  return mn_i





#
