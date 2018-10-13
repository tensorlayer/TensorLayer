#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMStateTuple

__all__ = [
    '_conv_linear',
    'initialize_rnn_state',
    'target_mask_op',
    'advanced_indexing_op',
    'retrieve_seq_length_op',
    'retrieve_seq_length_op2',
    'retrieve_seq_length_op3',
]


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:

    Parameters
    ----------
    args : tensor
        4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size : tuple of int
        Filter height and width.
    num_features : int
        Nnumber of features.
    bias_start : float
        Starting value to initialize the bias; 0 by default.
    scope : VariableScope
        For the created subgraph; defaults to "Linear".

    Returns
    --------
    - A 4D Tensor with shape [batch h w num_features]

    Raises
    -------
    - ValueError : if some of the arguments has unspecified or wrong shape.

    """
    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix",
            [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype
        )
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(args, 3), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias",
            [num_features], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
    return res + bias_term


def initialize_rnn_state(state, feed_dict=None):
    """Returns the initialized RNN state.
    The inputs are `LSTMStateTuple` or `State` of `RNNCells`, and an optional `feed_dict`.

    Parameters
    ----------
    state : RNN state.
        The TensorFlow's RNN state.
    feed_dict : dictionary
        Initial RNN state; if None, returns zero state.

    Returns
    -------
    RNN state
        The TensorFlow's RNN state.

    """
    if isinstance(state, LSTMStateTuple):
        c = state.c.eval(feed_dict=feed_dict)
        h = state.h.eval(feed_dict=feed_dict)
        return c, h

    else:
        new_state = state.eval(feed_dict=feed_dict)
        return new_state


def target_mask_op(data, pad_val=0):  # HangSheng: return tensor for mask,if input is tf.string
    """Return tensor for mask, if input is ``tf.string``."""
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.cast(tf.reduce_any(tf.not_equal(data, pad_val), axis=2), dtype=tf.int32)
    elif data_shape_size == 2:
        return tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32)
    elif data_shape_size == 1:
        raise ValueError("target_mask_op: data has wrong shape!")
    else:
        raise ValueError("target_mask_op: handling data_shape_size %s hasn't been implemented!" % (data_shape_size))


# Advanced Ops for Dynamic RNN
def advanced_indexing_op(inputs, index):
    """Advanced Indexing for Sequences, returns the outputs by given sequence lengths.
    When return the last output :class:`DynamicRNN` uses it to get the last outputs with the sequence lengths.

    Parameters
    -----------
    inputs : tensor for data
        With shape of [batch_size, n_step(max), n_features]
    index : tensor for indexing
        Sequence length in Dynamic RNN. [batch_size]

    Examples
    ---------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size, max_length, n_features = 3, 5, 2
    >>> z = np.random.uniform(low=-1, high=1, size=[batch_size, max_length, n_features]).astype(np.float32)
    >>> b_z = tf.constant(z)
    >>> sl = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    >>> o = advanced_indexing_op(b_z, sl)
    >>>
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>>
    >>> order = np.asarray([1,1,2])
    >>> print("real",z[0][order[0]-1], z[1][order[1]-1], z[2][order[2]-1])
    >>> y = sess.run([o], feed_dict={sl:order})
    >>> print("given",order)
    >>> print("out", y)
    real [-0.93021595  0.53820813] [-0.92548317 -0.77135968] [ 0.89952248  0.19149846]
    given [1 1 2]
    out [array([[-0.93021595,  0.53820813],
                [-0.92548317, -0.77135968],
                [ 0.89952248,  0.19149846]], dtype=float32)]

    References
    -----------
    - Modified from TFlearn (the original code is used for fixed length rnn), `references <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__.

    """
    batch_size = tf.shape(inputs)[0]
    # max_length = int(inputs.get_shape()[1])    # for fixed length rnn, length is given
    max_length = tf.shape(inputs)[1]  # for dynamic_rnn, length is unknown
    dim_size = int(inputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(inputs, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant


def retrieve_seq_length_op(data):
    """An op to compute the length of a sequence from input shape of [batch_size, n_step(max), n_features],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max), n_features] with zero padding on right hand side.

    Examples
    ---------
    >>> data = [[[1],[2],[0],[0],[0]],
    ...         [[1],[2],[3],[0],[0]],
    ...         [[1],[2],[6],[1],[0]]]
    >>> data = np.asarray(data)
    >>> print(data.shape)
    (3, 5, 1)
    >>> data = tf.constant(data)
    >>> sl = retrieve_seq_length_op(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> y = sl.eval()
    [2 3 4]

    Multiple features
    >>> data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
    ...         [[2,3],[2,4],[3,2],[0,0],[0,0]],
    ...         [[3,3],[2,2],[5,3],[1,2],[0,0]]]
    >>> print(sl)
    [4 3 4]

    References
    ------------
    Borrow from `TFlearn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__.

    """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), 2))
        length = tf.reduce_sum(used, 1)

        return tf.cast(length, tf.int32)


def retrieve_seq_length_op2(data):
    """An op to compute the length of a sequence, from input shape of [batch_size, n_step(max)],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] with zero padding on right hand side.

    Examples
    --------
    >>> data = [[1,2,0,0,0],
    ...         [1,2,3,0,0],
    ...         [1,2,6,1,0]]
    >>> o = retrieve_seq_length_op2(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> print(o.eval())
    [2 3 4]

    """
    return tf.reduce_sum(tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), 1)


def retrieve_seq_length_op3(data, pad_val=0):  # HangSheng: return tensor for sequence length, if input is tf.string
    """Return tensor for sequence length, if input is ``tf.string``."""
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.reduce_sum(tf.cast(tf.reduce_any(tf.not_equal(data, pad_val), axis=2), dtype=tf.int32), 1)
    elif data_shape_size == 2:
        return tf.reduce_sum(tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32), 1)
    elif data_shape_size == 1:
        raise ValueError("retrieve_seq_length_op3: data has wrong shape!")
    else:
        raise ValueError(
            "retrieve_seq_length_op3: handling data_shape_size %s hasn't been implemented!" % (data_shape_size)
        )
