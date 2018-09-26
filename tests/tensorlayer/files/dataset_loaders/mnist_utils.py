#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import gzip

import numpy as np

from tensorlayer import logging

from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ["_load_mnist_dataset"]


def _load_mnist_dataset(shape, path, name='mnist', url='http://yann.lecun.com/exdb/mnist/'):
    """A generic function to load mnist-like dataset.

    Parameters:
    ----------
    shape : tuple
        The shape of digit images.
    path : str
        The path that the data is downloaded to.
    name : str
        The dataset name you want to use(the default is 'mnist').
    url : str
        The url of dataset(the default is 'http://yann.lecun.com/exdb/mnist/').
    """
    path = os.path.join(path, name)

    # Define functions for loading mnist-like data's images and labels.
    # For convenience, they also download the requested files if needed.
    def load_mnist_images(path, filename):
        filepath = maybe_download_and_extract(filename, path, url)

        logging.info(filepath)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(shape)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(path, filename):
        filepath = maybe_download_and_extract(filename, path, url)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # Download and read the training and test set images and labels.
    logging.info("Load or Download {0} > {1}".format(name.upper(), path))
    X_train = load_mnist_images(path, 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(path, 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(path, 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(path, 't10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return X_train, y_train, X_val, y_val, X_test, y_test
