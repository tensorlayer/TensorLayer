#! /usr/bin/python
# -*- coding: utf-8 -*-

import numbers

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops

from tensorlayer import tl_logging as logging

__all__ = [
    'cross_entropy',
    'sigmoid_cross_entropy',
    'binary_cross_entropy',
    'mean_squared_error',
    'normalized_mean_square_error',
    'absolute_difference_error',
    'dice_coe',
    'dice_hard_coe',
    'iou_coe',
    'cross_entropy_seq',
    'cross_entropy_seq_with_mask',
    'cosine_similarity',
    'li_regularizer',
    'lo_regularizer',
    'maxnorm_regularizer',
    'maxnorm_o_regularizer',
    'maxnorm_i_regularizer',
]


def cross_entropy(output, target, name=None):
    """Softmax cross-entropy operation, returns the TensorFlow expression of cross-entropy for two distributions,
    it implements softmax internally. See ``tf.nn.sparse_softmax_cross_entropy_with_logits``.

    Parameters
    ----------
    output : Tensor
        A batch of distribution with shape: [batch_size, num of classes].
    target : Tensor
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.

    Examples
    --------
    >>> ce = tl.cost.cross_entropy(y_logits, y_target_logits, 'my_loss')

    References
    -----------
    - About cross-entropy: `<https://en.wikipedia.org/wiki/Cross_entropy>`__.
    - The code is borrowed from: `<https://en.wikipedia.org/wiki/Cross_entropy>`__.

    """
    if name is None:
        raise Exception("Please give a unique name to tl.cost.cross_entropy for TF1.0+")
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output, name=name))


def sigmoid_cross_entropy(output, target, name=None):
    """Sigmoid cross-entropy operation, see ``tf.nn.sigmoid_cross_entropy_with_logits``.

    Parameters
    ----------
    output : Tensor
        A batch of distribution with shape: [batch_size, num of classes].
    target : Tensor
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.

    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output, name=name))


def binary_cross_entropy(output, target, epsilon=1e-8, name='bce_loss'):
    """Binary cross entropy operation.

    Parameters
    ----------
    output : Tensor
        Tensor with type of `float32` or `float64`.
    target : Tensor
        The target distribution, format the same with `output`.
    epsilon : float
        A small value to avoid output to be zero.
    name : str
        An optional name to attach to this function.

    References
    -----------
    - `ericjang-DRAW <https://github.com/ericjang/draw/blob/master/draw.py#L73>`__

    """

    #     with ops.op_scope([output, target], name, "bce_loss") as name:
    #         output = ops.convert_to_tensor(output, name="preds")
    #         target = ops.convert_to_tensor(targets, name="target")

    with tf.name_scope(name):
        return tf.reduce_mean(
            tf.reduce_sum(-(target * tf.log(output + epsilon) + (1. - target) * tf.log(1. - output + epsilon)), axis=1)
        )

    # For brevity, let `x = output`, `z = target`.  The binary cross entropy loss is
    #
    #     loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))


def mean_squared_error(output, target, is_mean=False, name="mean_squared_error"):
    """Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : Tensor
        2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, height, width] or [batch_size, height, width, channel].
    target : Tensor
        The target distribution, format the same with `output`.
    is_mean : boolean
        Whether compute the mean or sum for each example.
            - If True, use ``tf.reduce_mean`` to compute the loss between one target and predict data.
            - If False, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`__

    """
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return mse


def normalized_mean_square_error(output, target):
    """Return the TensorFlow expression of normalized mean-square-error of two distributions.

    Parameters
    ----------
    output : Tensor
        2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, height, width] or [batch_size, height, width, channel].
    target : Tensor
        The target distribution, format the same with `output`.

    """
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=1))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(output, target), axis=[1, 2, 3]))
            nmse_b = tf.sqrt(tf.reduce_sum(tf.square(target), axis=[1, 2, 3]))
        nmse = tf.reduce_mean(nmse_a / nmse_b)
    return nmse


def absolute_difference_error(output, target, is_mean=False):
    """Return the TensorFlow expression of absolute difference error (L1) of two batch of data.

    Parameters
    ----------
    output : Tensor
        2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, height, width] or [batch_size, height, width, channel].
    target : Tensor
        The target distribution, format the same with `output`.
    is_mean : boolean
        Whether compute the mean or sum for each example.
            - If True, use ``tf.reduce_mean`` to compute the loss between one target and predict data.
            - If False, use ``tf.reduce_sum`` (default).

    """
    with tf.name_scope("mean_squared_error_loss"):
        if output.get_shape().ndims == 2:  # [batch_size, n_feature]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), 1))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), 1))
        elif output.get_shape().ndims == 3:  # [batch_size, w, h]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2]))
        elif output.get_shape().ndims == 4:  # [batch_size, w, h, c]
            if is_mean:
                loss = tf.reduce_mean(tf.reduce_mean(tf.abs(output - target), [1, 2, 3]))
            else:
                loss = tf.reduce_mean(tf.reduce_sum(tf.abs(output - target), [1, 2, 3]))
        else:
            raise Exception("Unknow dimension")
        return loss


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice


def dice_hard_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.

    Parameters
    -----------
    output : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    output = tf.cast(output > threshold, dtype=tf.float32)
    target = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(output, target), axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def iou_coe(output, target, threshold=0.5, axis=(1, 2, 3), smooth=1e-5):
    """Non-differentiable Intersection over Union (IoU) for comparing the
    similarity of two batch of data, usually be used for evaluating binary image segmentation.
    The coefficient between 0 to 1, and 1 means totally match.

    Parameters
    -----------
    output : tensor
        A batch of distribution with shape: [batch_size, ....], (any dimensions).
    target : tensor
        The target distribution, format the same with `output`.
    threshold : float
        The threshold value to be true.
    axis : tuple of integer
        All dimensions are reduced, default ``(1,2,3)``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.

    Notes
    ------
    - IoU cannot be used as training loss, people usually use dice coefficient for training, IoU and hard-dice for evaluating.

    """
    pre = tf.cast(output > threshold, dtype=tf.float32)
    truth = tf.cast(target > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(pre, truth), axis=axis)  # AND
    union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=axis)  # OR
    ## old axis=[0,1,2,3]
    # epsilon = 1e-5
    # batch_iou = inse / (union + epsilon)
    ## new haodong
    batch_iou = (inse + smooth) / (union + smooth)
    iou = tf.reduce_mean(batch_iou)
    return iou  #, pre, truth, inse, union


# ## test soft/hard dice and iou
# import numpy as np
# y = np.zeros((1,10,10,1))
# # y[0,0:5,0:5]=1.0
# o = np.zeros((1,10,10,1))
# # o[:,:,:,:] = 0            # what we want: dice=0   iou=0  OK
# # o[0,0:2,0:2]=0.3          # what we want: dice larger iou=0  OK
# # o[0,0:2,0:2]=0.6          # what we want: dice larger  iou small  OK
# # o[0,0:3,0:3]=0.6          # what we want: dice larger iou larger OK
# # o[0,0:3,0:3]=1            # what we want: dice larger iou same OK
# # o[0,0:5,0:5]=1            # what we want: dice=1 iou=1  OK
# # o[0,0:5,0:5]=0.3          # what we want: dice smaller  iou=0  OK
# # o[0,0:5,0:5]=1e-2           # what we want: dice≈0 iou=0  OK
# # o[0,8:10,8:10]=1.0        # what we want: dice=0 iou=0  OK
# # o[0,8:10,8:10]=1e-10        # what we want: dice=0 iou=0  OK
# # y[:,:,:,:] = o[:,:,:,:] = 0 # what we want: dice=1 iou=1  OK
# ## why in u-net, dice=1 hard-dice=1 iou=1 exist?? print bug?
#
# d = dice_coe(o, y, 'jaccard', smooth=1.)
# hd = dice_hard_coe(o, y, smooth=1e-5)
# i = iou_coe(o, y, smooth=1e-5)
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# # sess.run(tf.local_variables_initializer())
# print(sess.run([d,hd,i]))
# # p, t, i, u = sess.run([pre, truth, inse, union])
# # import pprint
# # pprint.pprint(((y>0.5)*(o>0.5)).astype(int).tolist())
# # pprint.pprint(p.tolist())
# # pprint.pprint(t.tolist())
# # pprint.pprint(i)
# # pprint.pprint(u)
# exit()


def cross_entropy_seq(logits, target_seqs, batch_size=None):  #, batch_size=1, num_steps=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for fixed length RNN outputs, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.

    Parameters
    ----------
    logits : Tensor
        2D tensor with shape of `[batch_size * n_steps, n_classes]`.
    target_seqs : Tensor
        The target sequence, 2D tensor `[batch_size, n_steps]`, if the number of step is dynamic, please use ``tl.cost.cross_entropy_seq_with_mask`` instead.
    batch_size : None or int.
        Whether to divide the cost by batch size.
            - If integer, the return cost will be divided by `batch_size`.
            - If None (default), the return cost will not be divided by anything.

    Examples
    --------
    >>> see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.for more details
    >>> input_data = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> targets = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> # build the network
    >>> print(net.outputs)
    (batch_size * n_steps, n_classes)
    >>> cost = tl.cost.cross_entropy_seq(network.outputs, targets)

    """
    sequence_loss_by_example_fn = tf.contrib.legacy_seq2seq.sequence_loss_by_example

    loss = sequence_loss_by_example_fn(
        [logits], [tf.reshape(target_seqs, [-1])], [tf.ones_like(tf.reshape(target_seqs, [-1]), dtype=tf.float32)]
    )
    # [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss)  #/ batch_size
    if batch_size is not None:
        cost = cost / batch_size
    return cost


def cross_entropy_seq_with_mask(logits, target_seqs, input_mask, return_details=False, name=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for Dynamic RNN with Synced sequence input and output.

    Parameters
    -----------
    logits : Tensor
        2D tensor with shape of [batch_size * ?, n_classes], `?` means dynamic IDs for each example.
        - Can be get from `DynamicRNNLayer` by setting ``return_seq_2d`` to `True`.
    target_seqs : Tensor
        int of tensor, like word ID. [batch_size, ?], `?` means dynamic IDs for each example.
    input_mask : Tensor
        The mask to compute loss, it has the same size with `target_seqs`, normally 0 or 1.
    return_details : boolean
        Whether to return detailed losses.
            - If False (default), only returns the loss.
            - If True, returns the loss, losses, weights and targets (see source code).

    Examples
    --------
    >>> batch_size = 64
    >>> vocab_size = 10000
    >>> embedding_size = 256
    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
    >>> target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target")
    >>> input_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="mask")
    >>> net = tl.layers.EmbeddingInputlayer(
    ...         inputs = input_seqs,
    ...         vocabulary_size = vocab_size,
    ...         embedding_size = embedding_size,
    ...         name = 'seq_embedding')
    >>> net = tl.layers.DynamicRNNLayer(net,
    ...         cell_fn = tf.contrib.rnn.BasicLSTMCell,
    ...         n_hidden = embedding_size,
    ...         dropout = (0.7 if is_train else None),
    ...         sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...         return_seq_2d = True,
    ...         name = 'dynamicrnn')
    >>> print(net.outputs)
    (?, 256)
    >>> net = tl.layers.DenseLayer(net, n_units=vocab_size, name="output")
    >>> print(net.outputs)
    (?, 10000)
    >>> loss = tl.cost.cross_entropy_seq_with_mask(net.outputs, target_seqs, input_mask)

    """
    targets = tf.reshape(target_seqs, [-1])  # to one vector
    weights = tf.to_float(tf.reshape(input_mask, [-1]))  # to one vector like targets
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name) * weights
    #losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name=name)) # for TF1.0 and others

    loss = tf.divide(
        tf.reduce_sum(losses),  # loss from mask. reduce_sum before element-wise mul with mask !!
        tf.reduce_sum(weights),
        name="seq_loss_with_mask"
    )

    if return_details:
        return loss, losses, weights, targets
    else:
        return loss


def cosine_similarity(v1, v2):
    """Cosine similarity [-1, 1].

    Parameters
    ----------
    v1, v2 : Tensor
        Tensor with the same shape [batch_size, n_feature].

    Returns
    -------
    Tensor
        a tensor of shape [batch_size].

    References
    ----------
    - `<https://en.wikipedia.org/wiki/Cosine_similarity>`__.

    """

    return tf.reduce_sum(tf.multiply(v1, v2), 1) / (
        tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1))
    )


## Regularization Functions
def li_regularizer(scale, scope=None):
    """Li regularization removes the neurons of previous layer. The `i` represents `inputs`.
    Returns a function that can be used to apply group li regularization to weights.
    The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`__.

    Parameters
    ----------
    scale : float
        A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: str
        An optional scope name for this function.

    Returns
    --------
    A function with signature `li(weights, name=None)` that apply Li regularization.

    Raises
    ------
    ValueError : if scale is outside of the range [0.0, 1.0] or if scale is not a float.

    """

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        if scale >= 1.:
            raise ValueError('Setting a scale greater than 1 on a regularizer: %g' % scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def li(weights):
        """Applies li regularization to weights."""
        with tf.name_scope('li_regularizer') as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            # if tf.__version__ <= '0.12':
            #     standard_ops_fn = standard_ops.mul
            # else:
            standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
                my_scale, standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
                name=scope
            )

    return li


def lo_regularizer(scale):
    """Lo regularization removes the neurons of current layer. The `o` represents `outputs`
    Returns a function that can be used to apply group lo regularization to weights.
    The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`__.

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

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)

    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        if scale >= 1.:
            raise ValueError('Setting a scale greater than 1 on a regularizer: %g' % scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def lo(weights, name='lo_regularizer'):
        """Applies group column regularization to weights."""
        with tf.name_scope(name) as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            # if tf.__version__ <= '0.12':
            #     standard_ops_fn = standard_ops.mul
            # else:
            standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
                my_scale, standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 0))),
                name=scope
            )

    return lo


def maxnorm_regularizer(scale=1.0):
    """Max-norm regularization returns a function that can be used to apply max-norm regularization to weights.

    More about max-norm, see `wiki-max norm <https://en.wikipedia.org/wiki/Matrix_norm#Max_norm>`_.
    The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`__.

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

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)

    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        # if scale >= 1.:
        #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
        #                    scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def mn(weights, name='max_regularizer'):
        """Applies max-norm regularization to weights."""
        with tf.name_scope(name) as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            #   if tf.__version__ <= '0.12':
            #       standard_ops_fn = standard_ops.mul
            #   else:
            standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(my_scale, standard_ops.reduce_max(standard_ops.abs(weights)), name=scope)

    return mn


def maxnorm_o_regularizer(scale):
    """Max-norm output regularization removes the neurons of current layer.
    Returns a function that can be used to apply max-norm regularization to each column of weight matrix.
    The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`__.

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

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)

    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        # if scale >= 1.:
        #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
        #                    scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def mn_o(weights, name='maxnorm_o_regularizer'):
        """Applies max-norm regularization to weights."""
        with tf.name_scope(name) as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            if tf.__version__ <= '0.12':
                standard_ops_fn = standard_ops.mul
            else:
                standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
                my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 0)), name=scope
            )

    return mn_o


def maxnorm_i_regularizer(scale):
    """Max-norm input regularization removes the neurons of previous layer.
    Returns a function that can be used to apply max-norm regularization to each row of weight matrix.
    The implementation follows `TensorFlow contrib <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py>`__.

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

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)

    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        # if scale >= 1.:
        #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
        #                    scale)
        if scale == 0.:
            logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def mn_i(weights, name='maxnorm_i_regularizer'):
        """Applies max-norm regularization to weights."""
        with tf.name_scope(name) as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            if tf.__version__ <= '0.12':
                standard_ops_fn = standard_ops.mul
            else:
                standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
                my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 1)), name=scope
            )

    return mn_i
