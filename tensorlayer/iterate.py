import numpy as np

def minibatches(inputs, targets, batchsize, shuffle=False):
    """
    The :function:`minibatches` return a dataset by
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
