#! /usr/bin/python
# -*- coding: utf-8 -*-

import gzip
import os

import numpy as np

import six.moves.cPickle as pickle
from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_imdb_dataset']


def load_imdb_dataset(
        path='data', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113, start_char=1, oov_char=2,
        index_from=3
):
    """Load IMDB dataset.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/imdb/``.
    nb_words : int
        Number of words to get.
    skip_top : int
        Top most frequent words to ignore (they will appear as oov_char value in the sequence data).
    maxlen : int
        Maximum sequence length. Any longer sequence will be truncated.
    seed : int
        Seed for reproducible data shuffling.
    start_char : int
        The start of a sequence will be marked with this character. Set to 1 because 0 is usually the padding character.
    oov_char : int
        Words that were cut out because of the num_words or skip_top limit will be replaced with this character.
    index_from : int
        Index actual words with this index and higher.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(
    ...                                 nb_words=20000, test_split=0.2)
    >>> print('X_train.shape', X_train.shape)
    (20000,)  [[1, 62, 74, ... 1033, 507, 27],[1, 60, 33, ... 13, 1053, 7]..]
    >>> print('y_train.shape', y_train.shape)
    (20000,)  [1 0 0 ..., 1 0 1]

    References
    -----------
    - `Modified from keras. <https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py>`__

    """
    path = os.path.join(path, 'imdb')

    filename = "imdb.pkl"
    url = 'https://s3.amazonaws.com/text-datasets/'
    maybe_download_and_extract(filename, path, url)

    if filename.endswith(".gz"):
        f = gzip.open(os.path.join(path, filename), 'rb')
    else:
        f = open(os.path.join(path, filename), 'rb')

    X, labels = pickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception(
            'After filtering for sequences shorter than maxlen=' + str(maxlen) + ', no sequence was kept. '
            'Increase maxlen.'
        )
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return X_train, y_train, X_test, y_test
