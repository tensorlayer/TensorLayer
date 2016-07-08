#! /usr/bin/python
# -*- coding: utf8 -*-




import tensorflow as tf
import os
from sys import platform as _platform
from tensorlayer.layers import set_keep
import collections
import random
import numpy as np

def generate_skip_gram_batch(data, batch_size, num_skips, skip_window, data_index=0):
    """Generate a training batch for the Skip-Gram model.

    Parameters
    ----------
    data : a list
        To present context.
    batch_size : an int
        Batch size to return.
    num_skips : an int
        How many times to reuse an input to generate a label.
    skip_window : an int
        How many words to consider left and right.
    data_index : an int
        Index of the context location.
        without using yield, this code use data_index to instead.

    Returns
    --------
    batch : a list
        Inputs
    labels : a list
        Labels
    data_index : an int
        Index of the context location.

    Example
    --------
    Setting num_skips=2, skip_window=1, use the right and left words.
    In the same way, num_skips=4, skip_window=2 means use the nearby 4 words.

    >>> data = [1,2,3,4,5,6,7,8,9,10,11]
    >>> batch, labels, data_index = tl.nlp.generate_skip_gram_batch(    \
        data=data, batch_size=8, num_skips=2, skip_window=1, data_index=0)
    >>> print(batch)
    ... [2 2 3 3 4 4 5 5]
    >>> print(labels)
    ... [[3]
    ... [1]
    ... [4]
    ... [2]
    ... [5]
    ... [3]
    ... [4]
    ... [6]]

    References
    -----------
    `TensorFlow word2vec tutorial <https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html#vector-representations-of-words>`_
    """
    # global data_index   # you can put data_index outside the function, then
    #       modify the global data_index in the function without return it.
    # note: without using yield, this code use data_index to instead.
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, data_index













#
