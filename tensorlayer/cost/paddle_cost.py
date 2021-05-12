#! /usr/bin/python
# -*- coding: utf-8 -*-

import paddle.nn.functional as F
import paddle as pd

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

def cross_entropy(output, target):
    """Softmax cross-entropy operation, returns the TensorFlow expression of cross-entropy for two distributions,
    it implements softmax internally. See ``tf.ops.sparse_softmax_cross_entropy_with_logits``.

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
    >>> import tensorlayer as tl
    >>> ce = tl.cost.cross_entropy(y_logits, y_target_logits)

    References
    -----------
    - About cross-entropy: `<https://en.wikipedia.org/wiki/Cross_entropy>`__.
    - The code is borrowed from: `<https://en.wikipedia.org/wiki/Cross_entropy>`__.

    """

    return F.cross_entropy(input=output, label=target)


def sigmoid_cross_entropy(output, target):
    """Sigmoid cross-entropy operation, see ``tf.ops.sigmoid_cross_entropy_with_logits``.

    Parameters
    ----------
    output : Tensor
        A batch of distribution with shape: [batch_size, num of classes].
    target : Tensor
        A batch of index with shape: [batch_size, ].
    name : string
        Name of this loss.

    """

    if output.shape[-1] == target.shape[-1]:
        pass
    else:
        depth = output.shape[-1]
        label = pd.fluid.layers.one_hot(target, depth=depth)
    out = pd.fluid.layers.sigmoid_cross_entropy_with_logits(x=output, label=label)
    out = pd.fluid.layers.reduce_mean(out)
    return out


def binary_cross_entropy(output, target, epsilon=1e-8):
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

    if output.shape[-1] == target.shape[-1]:
        pass
    else:
        depth = output.shape[-1]
        target = pd.fluid.layers.one_hot(target, depth=depth)
    out = pd.fluid.layers.reduce_sum(
            -(target * pd.log(output + epsilon) + (1. - target) * pd.log(1. - output + epsilon))
        )
    return out


def mean_squared_error(output, target, is_mean=False, axis=-1, name="mean_squared_error"):
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
    axis : int or list of int
        The dimensions to reduce.
    name : str
        An optional name to attach to this function.

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`__

    """

    if output.shape[-1] == target.shape[-1]:
        pass
    else:
        depth = output.shape[-1]
        target = pd.fluid.layers.one_hot(target, depth=depth)

    if is_mean:
        mse = F.mse_loss(input=output, label=target, reduction='mean')
    else:
        mse = F.mse_loss(input=output, label=target, reduction='sum')
    return mse


def normalized_mean_square_error(output, target, axis=-1, name="normalized_mean_squared_error_loss"):
    """Return the TensorFlow expression of normalized mean-square-error of two distributions.

    Parameters
    ----------
    output : Tensor
        2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, height, width] or [batch_size, height, width, channel].
    target : Tensor
        The target distribution, format the same with `output`.
    axis : int or list of int
        The dimensions to reduce.
    name : str
        An optional name to attach to this function.

    """

    if output.shape[-1] == target.shape[-1]:
        pass
    else:
        depth = output.shape[-1]
        target = pd.fluid.layers.one_hot(target, depth=depth)

    nmse_a = pd.sqrt(pd.fluid.layers.reduce_sum(pd.fluid.layers.square_error_cost(output, target), dim=axis))
    nmse_b = pd.sqrt(pd.fluid.layers.reduce_sum(pd.square(target), dim=axis))
    nmse = pd.fluid.layers.reduce_mean(nmse_a / nmse_b)
    return nmse


def absolute_difference_error(output, target, is_mean=False, axis=-1, name="absolute_difference_error_loss"):
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
    axis : int or list of int
        The dimensions to reduce.
    name : str
        An optional name to attach to this function.

    """


    if is_mean:
        loss = pd.fluid.layers.reduce_mean(pd.fluid.layers.reduce_mean(pd.abs(output - target), axis))
    else:
        loss = pd.fluid.layers.reduce_mean(pd.fluid.layers.reduce_sum(pd.abs(output - target), axis))
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
    >>> import tensorlayer as tl
    >>> outputs = tl.act.pixel_wise_softmax(outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """

    axis = list(axis)
    inse = pd.fluid.layers.reduce_sum(output * target, dim=axis)
    if loss_type == 'jaccard':
        l = pd.fluid.layers.reduce_sum(output * output, dim=axis)
        r = pd.fluid.layers.reduce_sum(target * target, dim=axis)
    elif loss_type == 'sorensen':
        l = pd.fluid.layers.reduce_sum(output, dim=axis)
        r = pd.fluid.layers.reduce_sum(target, dim=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = pd.fluid.layers.reduce_mean(dice)
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

    output = pd.cast(output > threshold, dtype='float32')
    target = pd.cast(target > threshold, dtype='float32')
    inse = pd.fluid.layers.reduce_sum(pd.multiply(output, target), dim=list(axis))
    l = pd.fluid.layers.reduce_sum(output, dim=list(axis))
    r = pd.fluid.layers.reduce_sum(target, dim=list(axis))

    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = pd.fluid.layers.reduce_mean(hard_dice)
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

    pre = pd.cast(output > threshold, dtype='float32')
    truth = pd.cast(target > threshold, dtype='float32')
    inse = pd.fluid.layers.reduce_sum(pd.multiply(pre, truth), dim=axis)  # AND
    union = pd.fluid.layers.reduce_sum(pd.cast(pd.add(pre, truth) >= 1, dtype='float32'), dim=axis)  # OR
    batch_iou = (inse + smooth) / (union + smooth)
    iou = pd.fluid.layers.reduce_mean(batch_iou, name='iou_coe')
    return iou


def sequence_loss_by_example(
    logits, targets, weights, average_across_timesteps=True, softmax_loss_function=None, name=None
):
    """Weighted cross-entropy loss for a sequence of logits (per example). see original tensorflow code :
    <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L1057>

    Parameters
    ----------
    logits: List
        List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List
        List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List
        List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: Boolean
        If set, divide the returned cost by the total label weight.
    softmax_loss_function: None or Function
        Function (labels, logits) -> loss-batch to be used instead of the standard softmax (the default if this is None).
        **Note that to avoid confusion, it is required for the function to accept named arguments.**
    name: None or str
        Optional name for this operation, default: "sequence_loss_by_example".

    Returns
    -------
    1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises
    ------
    ValueError: If len(logits) is different from len(targets) or len(weights).

    """

    raise NotImplementedError("Not Implemented.")


def cross_entropy_seq(logits, target_seqs, batch_size=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for fixed length RNN outputs, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py>`__.

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
    >>> import tensorlayer as tl
    >>> # see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py>`__.for more details
    >>> # outputs shape : (batch_size * n_steps, n_classes)
    >>> # targets shape : (batch_size, n_steps)
    >>> cost = tl.cost.cross_entropy_seq(outputs, targets)

    """

    raise NotImplementedError("Not Implemented.")


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
    >>> import tensorlayer as tl
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> batch_size = 64
    >>> vocab_size = 10000
    >>> embedding_size = 256
    >>> ni = tl.layers.Input([batch_size, None], dtype=tf.int64)
    >>> net = tl.layers.Embedding(
    ...         vocabulary_size = vocab_size,
    ...         embedding_size = embedding_size,
    ...         name = 'seq_embedding')(ni)
    >>> net = tl.layers.RNN(
    ...         cell =tf.keras.layers.LSTMCell(units=embedding_size, dropout=0.1),
    ...         return_seq_2d = True,
    ...         name = 'dynamicrnn')(net)
    >>> net = tl.layers.Dense(n_units=vocab_size, name="output")(net)
    >>> model = tl.models.Model(inputs=ni, outputs=net)
    >>> input_seqs = np.random.randint(0, 10, size=(batch_size, 10), dtype=np.int64)
    >>> target_seqs = np.random.randint(0, 10, size=(batch_size, 10), dtype=np.int64)
    >>> input_mask = np.random.randint(0, 2, size=(batch_size, 10), dtype=np.int64)
    >>> outputs = model(input_seqs, is_train=True)
    >>> loss = tl.cost.cross_entropy_seq_with_mask(outputs, target_seqs, input_mask)

    """

    raise NotImplementedError("Not Implemented.")


def cosine_similarity(v1, v2):
    """Cosine similarity [-1, 1].

    Parameters
    ----------
    v1, v2 : Tensor
        Tensor with the same shape [batch_size, n_feature].

    References
    ----------
    - `Wiki <https://en.wikipedia.org/wiki/Cosine_similarity>`__.

    """

    return pd.fluid.layers.reduce_sum(pd.multiply(v1, v2), 1) / \
        (pd.sqrt(pd.fluid.layers.reduce_sum(pd.multiply(v1, v1), 1)) *
         pd.sqrt(pd.fluid.layers.reduce_sum(pd.multiply(v2, v2), 1)))


# Regularization Functions
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

    raise NotImplementedError("Not Implemented.")


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

    raise NotImplementedError("Not Implemented.")


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

    raise NotImplementedError("Not Implemented.")


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

    raise NotImplementedError("Not Implemented.")


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

    raise NotImplementedError("Not Implemented.")


def huber_loss(
    output, target, is_mean=True, delta=1.0, dynamichuber=False, reverse=False, axis=-1, epsilon=0.00001, name=None
):
    """Huber Loss operation, see ``https://en.wikipedia.org/wiki/Huber_loss`` .
    Reverse Huber Loss operation, see  ''https://statweb.stanford.edu/~owen/reports/hhu.pdf''.
    Dynamic Reverse Huber Loss operation, see  ''https://arxiv.org/pdf/1606.00373.pdf''.

    Parameters
    ----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    is_mean : boolean
        Whether compute the mean or sum for each example.
        - If True, use ``tf.reduce_mean`` to compute the loss between one target and predict data (default).
        - If False, use ``tf.reduce_sum``.
    delta: float
        The point where the huber loss function changes from a quadratic to linear.
    dynamichuber: boolean
        Whether compute the coefficient c for each batch.
        - If True, c is 20% of the maximal per-batch error.
        - If False, c is delta.
    reverse: boolean
        Whether compute the reverse huber loss.
    axis : int or list of int
        The dimensions to reduce.
    epsilon:
        Eplison.
    name : string
        Name of this loss.

    """

    raise NotImplementedError("Not Implemented.")