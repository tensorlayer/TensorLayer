#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from six.moves import xrange

__all__ = [
    'minibatches',
    'seq_minibatches',
    'seq_minibatches2',
    'ptb_iterator',
]


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batch size.

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    targets : numpy.array
        The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=2, shuffle=False):
    >>>     print(batch)
    (array([['a', 'a'], ['b', 'b']], dtype='<U1'), array([0, 1]))
    (array([['c', 'c'], ['d', 'd']], dtype='<U1'), array([2, 3]))
    (array([['e', 'e'], ['f', 'f']], dtype='<U1'), array([4, 5]))

    Notes
    -----
    If you have two inputs and one label and want to shuffle them together, e.g. X1 (1000, 100), X2 (1000, 80) and Y (1000, 1), you can stack them together (`np.hstack((X1, X2))`)
    into (1000, 180) and feed to ``inputs``. After getting a batch, you can split it back into X1 and X2.

    """

    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if (isinstance(inputs, list) or isinstance(targets, list)) and (shuffle ==True):
            # zsdonghao: for list indexing when shuffle==True
            yield [inputs[i] for i in excerpt], [targets[i] for i in excerpt]
        else:
            yield inputs[excerpt], targets[excerpt]


def seq_minibatches(inputs, targets, batch_size, seq_length, stride=1):
    """Generate a generator that return a batch of sequence inputs and targets.
    If `batch_size=100` and `seq_length=5`, one return will have 500 rows (examples).

    Parameters
    ----------
    inputs : numpy.array
        The input features, every row is a example.
    targets : numpy.array
        The labels of inputs, every element is a example.
    batch_size : int
        The batch size.
    seq_length : int
        The sequence length.
    stride : int
        The stride step, default is 1.

    Examples
    --------
    Synced sequence input and output.

    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0, 1, 2, 3, 4, 5])
    >>> for batch in tl.iterate.seq_minibatches(inputs=X, targets=y, batch_size=2, seq_length=2, stride=1):
    >>>     print(batch)
    (array([['a', 'a'], ['b', 'b'], ['b', 'b'], ['c', 'c']], dtype='<U1'), array([0, 1, 1, 2]))
    (array([['c', 'c'], ['d', 'd'], ['d', 'd'], ['e', 'e']], dtype='<U1'), array([2, 3, 3, 4]))

    Many to One

    >>> return_last = True
    >>> num_steps = 2
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> Y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.seq_minibatches(inputs=X, targets=Y, batch_size=2, seq_length=num_steps, stride=1):
    >>>     x, y = batch
    >>>     if return_last:
    >>>         tmp_y = y.reshape((-1, num_steps) + y.shape[1:])
    >>>     y = tmp_y[:, -1]
    >>>     print(x, y)
    [['a' 'a']
    ['b' 'b']
    ['b' 'b']
    ['c' 'c']] [1 2]
    [['c' 'c']
    ['d' 'd']
    ['d' 'd']
    ['e' 'e']] [3 4]

    """

    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    n_loads = (batch_size * stride) + (seq_length - stride)

    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:], dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:], dtype=targets.dtype)
        for b_idx in xrange(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1, ) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1, ) + targets.shape[1:])
        yield flatten_inputs, flatten_targets


def seq_minibatches2(inputs, targets, batch_size, num_steps):
    """Generate a generator that iterates on two list of words. Yields (Returns) the source contexts and
    the target context by the given batch_size and num_steps (sequence_length).
    In TensorFlow's tutorial, this generates the `batch_size` pointers into the raw PTB data, and allows minibatch iteration along these pointers.

    Parameters
    ----------
    inputs : list of data
        The context in list format; note that context usually be represented by splitting by space, and then convert to unique word IDs.
    targets : list of data
        The context in list format; note that context usually be represented by splitting by space, and then convert to unique word IDs.
    batch_size : int
        The batch size.
    num_steps : int
        The number of unrolls. i.e. sequence length

    Yields
    ------
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].

    Raises
    ------
    ValueError : if batch_size or num_steps are too high.

    Examples
    --------
    >>> X = [i for i in range(20)]
    >>> Y = [i for i in range(20,40)]
    >>> for batch in tl.iterate.seq_minibatches2(X, Y, batch_size=2, num_steps=3):
    ...     x, y = batch
    ...     print(x, y)

    [[  0.   1.   2.]
    [ 10.  11.  12.]]
    [[ 20.  21.  22.]
    [ 30.  31.  32.]]

    [[  3.   4.   5.]
    [ 13.  14.  15.]]
    [[ 23.  24.  25.]
    [ 33.  34.  35.]]

    [[  6.   7.   8.]
    [ 16.  17.  18.]]
    [[ 26.  27.  28.]
    [ 36.  37.  38.]]

    Notes
    -----
    - Hint, if the input data are images, you can modify the source code `data = np.zeros([batch_size, batch_len)` to `data = np.zeros([batch_size, batch_len, inputs.shape[1], inputs.shape[2], inputs.shape[3]])`.
    """

    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    data_len = len(inputs)
    batch_len = data_len // batch_size
    # data = np.zeros([batch_size, batch_len])
    data = np.zeros((batch_size, batch_len) + inputs.shape[1:], dtype=inputs.dtype)
    data2 = np.zeros([batch_size, batch_len])

    for i in range(batch_size):
        data[i] = inputs[batch_len * i:batch_len * (i + 1)]
        data2[i] = targets[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        x2 = data2[:, i * num_steps:(i + 1) * num_steps]
        yield (x, x2)


def ptb_iterator(raw_data, batch_size, num_steps):
    """Generate a generator that iterates on a list of words, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.
    Yields the source contexts and the target context by the given batch_size and num_steps (sequence_length).

    In TensorFlow's tutorial, this generates `batch_size` pointers into the raw
    PTB data, and allows minibatch iteration along these pointers.

    Parameters
    ----------
    raw_data : a list
            the context in list format; note that context usually be
            represented by splitting by space, and then convert to unique
            word IDs.
    batch_size : int
            the batch size.
    num_steps : int
            the number of unrolls. i.e. sequence_length

    Yields
    ------
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

    Raises
    ------
    ValueError : if batch_size or num_steps are too high.

    Examples
    --------
    >>> train_data = [i for i in range(20)]
    >>> for batch in tl.iterate.ptb_iterator(train_data, batch_size=2, num_steps=3):
    >>>     x, y = batch
    >>>     print(x, y)
    [[ 0  1  2] <---x                       1st subset/ iteration
     [10 11 12]]
    [[ 1  2  3] <---y
     [11 12 13]]

    [[ 3  4  5]  <--- 1st batch input       2nd subset/ iteration
     [13 14 15]] <--- 2nd batch input
    [[ 4  5  6]  <--- 1st batch target
     [14 15 16]] <--- 2nd batch target

    [[ 6  7  8]                             3rd subset/ iteration
     [16 17 18]]
    [[ 7  8  9]
     [17 18 19]]
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
