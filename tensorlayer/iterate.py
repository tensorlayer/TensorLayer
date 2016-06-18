import numpy as np

def minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Input a group of example in 2D numpy.array and their labels.
    Return the examples and labels by the given batchsize

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batchsize : int
        The batch size.
    shuffle : True, False
        If True, shuffle the dataset before return.

    Examples
    --------
    >>> xxx
    >>> xxx
    """
    assert len(inputs) == len(targets)
    if shuffle: # 打乱顺序
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def minibatches_for_sequence2D(inputs, targets, batch_size, sequence_length, stride=1):
    """
    Input a group of example in 2D numpy.array and their labels.
    Return the examples and labels by the given batchsize, sequence_length.
    Use for RNN.

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batchsize : int
        The batch size must be a multiple of sequence_length: int(batch_size % sequence_length) == 0
    sequence_length : int
        The sequence length
    stride: int
        The stride step

    Examples
    --------
    >>> sequence_length = 2
    >>> batch_size = 4
    >>> stride = 1
    >>> X_train = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
    >>> y_train = np.asarray(['0','1','2','3','4','5','6','7'])
    >>> print('X_train =\n %s' % X_train)
    >>> print('y_train =\n %s' % y_train)
    >>> for batch in minibatches_for_sequence2D(X_train, y_train, batch_size=batch_size, sequence_length=sequence_length, stride=stride):
    >>>     inputs, targets = batch
    >>>     print(inputs)
    >>>     print(targets)
    ... [[ 1.  2.  3.]
    ... [ 4.  5.  6.]
    ... [ 4.  5.  6.]
    ... [ 7.  8.  9.]]
    ... [1 2]
    ... [[  4.   5.   6.]
    ... [  7.   8.   9.]
    ... [  7.   8.   9.]
    ... [ 10.  11.  12.]]
    ... [2 3]
    ... ...
    ... [[ 16.  17.  18.]
    ... [ 19.  20.  21.]
    ... [ 19.  20.  21.]
    ... [ 22.  23.  24.]]
    ... [6 7]
    """
    print('len(targets)=%d batch_size=%d sequence_length=%d stride=%d' % (len(targets), batch_size, sequence_length, stride))
    assert len(inputs) == len(targets), '1 feature vector have 1 target vector/value' #* sequence_length
    # assert int(batch_size % sequence_length) == 0, 'batch_size % sequence_length must == 0\
    # batch_size is number of examples rather than number of targets'

    # print(inputs.shape, len(inputs), len(inputs[0]))

    n_targets = int(batch_size/sequence_length)
    # n_targets = int(np.ceil(batch_size/sequence_length))
    X = np.empty(shape=(0,len(inputs[0])), dtype=np.float32)
    y = np.zeros(shape=(1, n_targets), dtype=np.int32)

    for idx in range(sequence_length, len(inputs), stride):  # go through all example during 1 epoch
        for n in range(n_targets):   # for num of target
            X = np.concatenate((X, inputs[idx-sequence_length+n:idx+n]))
            y[0][n] = targets[idx-1+n]
            # y = np.vstack((y, targets[idx-1+n]))
        yield X, y[0]
        X = np.empty(shape=(0,len(inputs[0])))
        # y = np.empty(shape=(1,0))


def minibatches_for_sequence4D(inputs, targets, batch_size, sequence_length, stride=1): #
    """
    Input a group of example in 4D numpy.array and their labels.
    Return the examples and labels by the given batchsize, sequence_length.
    Use for RNN.

    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batchsize : int
        The batch size must be a multiple of sequence_length: int(batch_size % sequence_length) == 0
    sequence_length : int
        The sequence length
    stride: int
        The stride step

    Examples
    --------
    >>> sequence_length = 2
    >>> batch_size = 2
    >>> stride = 1
    >>> X_train = np.asarray([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[22,23,24]])
    >>> y_train = np.asarray(['0','1','2','3','4','5','6','7'])
    >>> X_train = np.expand_dims(X_train, axis=1)
    >>> X_train = np.expand_dims(X_train, axis=3)
    >>> for batch in minibatches_for_sequence4D(X_train, y_train, batch_size=batch_size, sequence_length=sequence_length, stride=stride):
    >>>     inputs, targets = batch
    >>>     print(inputs)
    >>>     print(targets)
    ... [[[[ 1.]
    ...    [ 2.]
    ...    [ 3.]]]
    ... [[[ 4.]
    ...   [ 5.]
    ...   [ 6.]]]]
    ... [1]
    ... [[[[ 4.]
    ...    [ 5.]
    ...    [ 6.]]]
    ... [[[ 7.]
    ...   [ 8.]
    ...   [ 9.]]]]
    ... [2]
    ... ...
    ... [[[[ 19.]
    ...    [ 20.]
    ...    [ 21.]]]
    ... [[[ 22.]
    ...   [ 23.]
    ...   [ 24.]]]]
    ... [7]
    """
    print('len(targets)=%d batch_size=%d sequence_length=%d stride=%d' % (len(targets), batch_size, sequence_length, stride))
    assert len(inputs) == len(targets), '1 feature vector have 1 target vector/value' #* sequence_length
    # assert int(batch_size % sequence_length) == 0, 'in LSTM, batch_size % sequence_length must == 0\
    # batch_size is number of X_train rather than number of targets'
    assert stride >= 1, 'stride must be >=1, at least move 1 step for each iternation'

    n_example, n_channels, width, height = inputs.shape
    print('n_example=%d n_channels=%d width=%d height=%d' % (n_example, n_channels, width, height))

    n_targets = int(np.ceil(batch_size/sequence_length)) # 实际为 batchsize/sequence_length + 1
    print(n_targets)
    X = np.zeros(shape=(batch_size, n_channels, width, height), dtype=np.float32)
    # X = np.zeros(shape=(n_targets, sequence_length, n_channels, width, height), dtype=np.float32)
    y = np.zeros(shape=(1,n_targets), dtype=np.int32)
    # y = np.empty(shape=(0,1), dtype=np.float32)
    # time.sleep(2)
    for idx in range(sequence_length, n_example-n_targets+2, stride):  # go through all example during 1 epoch
        for n in range(n_targets):   # for num of target
            # print(idx+n, inputs[idx-sequence_length+n : idx+n].shape)
            X[n*sequence_length : (n+1)*sequence_length] = inputs[idx+n-sequence_length : idx+n]
            # X[n] = inputs[idx-sequence_length+n:idx+n]
            y[0][n] = targets[idx+n-1]
            # y = np.vstack((y, targets[idx-1+n]))
        # y = targets[idx: idx+n_targets]
        yield X, y[0]
