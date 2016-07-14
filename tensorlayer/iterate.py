#! /usr/bin/python
# -*- coding: utf8 -*-



import numpy as np
from six.moves import xrange

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """
    Generate a generator that input a group of example in 2D numpy.array and
    their labels, return the examples and labels by the given batchsize.

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
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
    ... (array([['a', 'a'],
    ...        ['b', 'b']],
    ...         dtype='<U1'), array([0, 1]))
    ... (array([['c', 'c'],
    ...        ['d', 'd']],
    ...         dtype='<U1'), array([2, 3]))
    ... (array([['e', 'e'],
    ...        ['f', 'f']],
    ...         dtype='<U1'), array([4, 5]))
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def seq_minibatches(inputs, targets, batch_size, seq_length, stride=1):
    """
    Generate a generator that return a batch of sequence inputs and targets.
    It is for the "Synced sequence input and output" as
    `Karpathy RNN Blog <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_
    shows.

    If batch_size = 100, seq_length = 5, a return input will have 500 rows (examples).

    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.seq_minibatches(inputs=X, targets=y, batch_size=2, seq_length=2, stride=1):
    >>>     print(batch)
    ... (array([['a', 'a'],
    ...        ['b', 'b'],
    ...         ['b', 'b'],
    ...         ['c', 'c']],
    ...         dtype='<U1'), array([0, 1, 1, 2]))
    ... (array([['c', 'c'],
    ...         ['d', 'd'],
    ...         ['d', 'd'],
    ...         ['e', 'e']],
    ...         dtype='<U1'), array([2, 3, 3, 4]))
    """
    assert len(inputs) == len(targets)
    n_loads = (batch_size * stride) + (seq_length - stride)
    for start_idx in range(0, len(inputs) - n_loads + 1, (batch_size * stride)):
        seq_inputs = np.zeros((batch_size, seq_length) + inputs.shape[1:],
                              dtype=inputs.dtype)
        seq_targets = np.zeros((batch_size, seq_length) + targets.shape[1:],
                               dtype=targets.dtype)
        for b_idx in xrange(batch_size):
            start_seq_idx = start_idx + (b_idx * stride)
            end_seq_idx = start_seq_idx + seq_length
            seq_inputs[b_idx] = inputs[start_seq_idx:end_seq_idx]
            seq_targets[b_idx] = targets[start_seq_idx:end_seq_idx]
        flatten_inputs = seq_inputs.reshape((-1,) + inputs.shape[1:])
        flatten_targets = seq_targets.reshape((-1,) + targets.shape[1:])
        yield flatten_inputs, flatten_targets


# def minibatches_for_sequence2D(inputs, targets, batch_size, sequence_length, stride=1):
#     """
#     Input a group of example in 2D numpy.array and their labels.
#     Return the examples and labels by the given batchsize, sequence_length.
#     Use for RNN.
#
#     Parameters
#     ----------
#     inputs : numpy.array
#         (X) The input features, every row is a example.
#     targets : numpy.array
#         (y) The labels of inputs, every row is a example.
#     batchsize : int
#         The batch size must be a multiple of sequence_length: int(batch_size % sequence_length) == 0
#     sequence_length : int
#         The sequence length
#     stride : int
#         The stride step
#
#     Examples
#     --------
#     >>> sequence_length = 2
#     >>> batch_size = 4
#     >>> stride = 1
#     >>> X_train = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
#     >>> y_train = np.asarray(['0','1','2','3','4','5','6','7'])
#     >>> print('X_train = %s' % X_train)
#     >>> print('y_train = %s' % y_train)
#     >>> for batch in minibatches_for_sequence2D(X_train, y_train, batch_size=batch_size, sequence_length=sequence_length, stride=stride):
#     >>>     inputs, targets = batch
#     >>>     print(inputs)
#     >>>     print(targets)
#     ... [[ 1.  2.  3.]
#     ... [ 4.  5.  6.]
#     ... [ 4.  5.  6.]
#     ... [ 7.  8.  9.]]
#     ... [1 2]
#     ... [[  4.   5.   6.]
#     ... [  7.   8.   9.]
#     ... [  7.   8.   9.]
#     ... [ 10.  11.  12.]]
#     ... [2 3]
#     ... ...
#     ... [[ 16.  17.  18.]
#     ... [ 19.  20.  21.]
#     ... [ 19.  20.  21.]
#     ... [ 22.  23.  24.]]
#     ... [6 7]
#     """
#     print('len(targets)=%d batch_size=%d sequence_length=%d stride=%d' % (len(targets), batch_size, sequence_length, stride))
#     assert len(inputs) == len(targets), '1 feature vector have 1 target vector/value' #* sequence_length
#     # assert int(batch_size % sequence_length) == 0, 'batch_size % sequence_length must == 0\
#     # batch_size is number of examples rather than number of targets'
#
#     # print(inputs.shape, len(inputs), len(inputs[0]))
#
#     n_targets = int(batch_size/sequence_length)
#     # n_targets = int(np.ceil(batch_size/sequence_length))
#     X = np.empty(shape=(0,len(inputs[0])), dtype=np.float32)
#     y = np.zeros(shape=(1, n_targets), dtype=np.int32)
#
#     for idx in range(sequence_length, len(inputs), stride):  # go through all example during 1 epoch
#         for n in range(n_targets):   # for num of target
#             X = np.concatenate((X, inputs[idx-sequence_length+n:idx+n]))
#             y[0][n] = targets[idx-1+n]
#             # y = np.vstack((y, targets[idx-1+n]))
#         yield X, y[0]
#         X = np.empty(shape=(0,len(inputs[0])))
#         # y = np.empty(shape=(1,0))
#
#
# def minibatches_for_sequence4D(inputs, targets, batch_size, sequence_length, stride=1): #
#     """
#     Input a group of example in 4D numpy.array and their labels.
#     Return the examples and labels by the given batchsize, sequence_length.
#     Use for RNN.
#
#     Parameters
#     ----------
#     inputs : numpy.array
#         (X) The input features, every row is a example.
#     targets : numpy.array
#         (y) The labels of inputs, every row is a example.
#     batchsize : int
#         The batch size must be a multiple of sequence_length: int(batch_size % sequence_length) == 0
#     sequence_length : int
#         The sequence length
#     stride : int
#         The stride step
#
#     Examples
#     --------
#     >>> sequence_length = 2
#     >>> batch_size = 2
#     >>> stride = 1
#     >>> X_train = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
#     >>> y_train = np.asarray(['0','1','2','3','4','5','6','7'])
#     >>> X_train = np.expand_dims(X_train, axis=1)
#     >>> X_train = np.expand_dims(X_train, axis=3)
#     >>> for batch in minibatches_for_sequence4D(X_train, y_train, batch_size=batch_size, sequence_length=sequence_length, stride=stride):
#     >>>     inputs, targets = batch
#     >>>     print(inputs)
#     >>>     print(targets)
#     ... [[[[ 1.]
#     ...    [ 2.]
#     ...    [ 3.]]]
#     ... [[[ 4.]
#     ...   [ 5.]
#     ...   [ 6.]]]]
#     ... [1]
#     ... [[[[ 4.]
#     ...    [ 5.]
#     ...    [ 6.]]]
#     ... [[[ 7.]
#     ...   [ 8.]
#     ...   [ 9.]]]]
#     ... [2]
#     ... ...
#     ... [[[[ 19.]
#     ...    [ 20.]
#     ...    [ 21.]]]
#     ... [[[ 22.]
#     ...   [ 23.]
#     ...   [ 24.]]]]
#     ... [7]
#     """
#     print('len(targets)=%d batch_size=%d sequence_length=%d stride=%d' % (len(targets), batch_size, sequence_length, stride))
#     assert len(inputs) == len(targets), '1 feature vector have 1 target vector/value' #* sequence_length
#     # assert int(batch_size % sequence_length) == 0, 'in LSTM, batch_size % sequence_length must == 0\
#     # batch_size is number of X_train rather than number of targets'
#     assert stride >= 1, 'stride must be >=1, at least move 1 step for each iternation'
#
#     n_example, n_channels, width, height = inputs.shape
#     print('n_example=%d n_channels=%d width=%d height=%d' % (n_example, n_channels, width, height))
#
#     n_targets = int(np.ceil(batch_size/sequence_length)) # 实际为 batchsize/sequence_length + 1
#     print(n_targets)
#     X = np.zeros(shape=(batch_size, n_channels, width, height), dtype=np.float32)
#     # X = np.zeros(shape=(n_targets, sequence_length, n_channels, width, height), dtype=np.float32)
#     y = np.zeros(shape=(1,n_targets), dtype=np.int32)
#     # y = np.empty(shape=(0,1), dtype=np.float32)
#     # time.sleep(2)
#     for idx in range(sequence_length, n_example-n_targets+2, stride):  # go through all example during 1 epoch
#         for n in range(n_targets):   # for num of target
#             # print(idx+n, inputs[idx-sequence_length+n : idx+n].shape)
#             X[n*sequence_length : (n+1)*sequence_length] = inputs[idx+n-sequence_length : idx+n]
#             # X[n] = inputs[idx-sequence_length+n:idx+n]
#             y[0][n] = targets[idx+n-1]
#             # y = np.vstack((y, targets[idx-1+n]))
#         # y = targets[idx: idx+n_targets]
#         yield X, y[0]
