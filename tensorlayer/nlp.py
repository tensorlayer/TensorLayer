#! /usr/bin/python
# -*- coding: utf8 -*-




import tensorflow as tf
import os
from sys import platform as _platform
from .layers import set_keep
import collections
import random
import numpy as np
import warnings

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




def sample(a, temperature=1.0):
    """Sample an index from a probability array.

    Parameters
    ----------
    a : a list
        List of probabilities.
    temperature : float or None
        The higher the more uniform.
        When a = [0.1, 0.2, 0.7],
            temperature = 0.7, the distribution will be sharpen [ 0.05048273  0.13588945  0.81362782]
            temperature = 1.0, the distribution will be the same [0.1    0.2    0.7]
            temperature = 1.5, the distribution will be filtered [ 0.16008435  0.25411807  0.58579758]
        If None, it will be ``np.argmax(a)``

    Note
    ------
    No matter what is the temperature and input list, the sum of all probabilities will be one.
    Even if input list = [1, 100, 200], the sum of all probabilities will still be one.

    For large vocabulary_size, choice a higher temperature to avoid error.
    """
    b = np.copy(a)
    try:
        if temperature == 1:
            return np.argmax(np.random.multinomial(1, a, 1))
        if temperature is None:
            return np.argmax(a)
        else:
            a = np.log(a) / temperature
            a = np.exp(a) / np.sum(np.exp(a))
            return np.argmax(np.random.multinomial(1, a, 1))
    except:
        # np.set_printoptions(threshold=np.nan)
        # print(a)
        # print(np.sum(a))
        # print(np.max(a))
        # print(np.min(a))
        # exit()
        message = "For large vocabulary_size, choice a higher temperature\
         to avoid log error. Hint : use ``sample_top``. "
        warnings.warn(message, Warning)
        # print(a)
        # print(b)
        return np.argmax(np.random.multinomial(1, b, 1))



def sample_top(a, top_k=10):
    """Sample from ``top_k`` probabilities.
    """
    a = np.array(a)
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # a = a[idx]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice



#
