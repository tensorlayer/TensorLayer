#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import pickle

import numpy as np

from tensorlayer import logging

from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_cifar10_dataset']


def load_cifar10_dataset(shape=(-1, 32, 32, 3), path='data'):
    """Load CIFAR-10 dataset.

    It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with
    10000 images. The test batch contains exactly 1000 randomly-selected images from
    each class. The training batches contain the remaining images in random order,
    but some training batches may contain more images from one class than another.
    Between them, the training batches contain exactly 5000 images from each class.

    Parameters
    ----------
    shape : tupe
        The shape of digit images e.g. (-1, 3, 32, 32) and (-1, 32, 32, 3).
    path : str
        The path that the data is downloaded to, defaults is ``data/cifar10/``.
    plotable : boolean
        Whether to plot some image examples, False as default.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))

    References
    ----------
    - `CIFAR website <https://www.cs.toronto.edu/~kriz/cifar.html>`__
    - `Data download link <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`__
    - `<https://teratail.com/questions/28932>`__

    """
    path = os.path.join(path, 'cifar10')
    logging.info("Load or Download cifar10 > {}".format(path))

    # Helper function to unpickle the data
    def unpickle(file):
        fp = open(file, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        else:
            raise RuntimeError("Sys Version Unsupported")
        fp.close()
        return data

    filename = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/'
    # Download and uncompress file
    maybe_download_and_extract(filename, path, url, extract=True)

    # Unpickle file and fill in data
    X_train = None
    y_train = []
    for i in range(1, 6):
        data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', "data_batch_{}".format(i)))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', "test_batch"))
    X_test = test_data_dic['data']
    y_test = np.array(test_data_dic['labels'])

    if shape == (-1, 3, 32, 32):
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)
    elif shape == (-1, 32, 32, 3):
        X_test = X_test.reshape(shape, order='F')
        X_train = X_train.reshape(shape, order='F')
        X_test = np.transpose(X_test, (0, 2, 1, 3))
        X_train = np.transpose(X_train, (0, 2, 1, 3))
    else:
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)

    y_train = np.array(y_train)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test
