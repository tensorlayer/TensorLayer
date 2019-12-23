import logging
import os
import numpy as np

from ..base import Dataset
from .mnist import _load_mnist_images, _load_mnist_labels

__all__ = ['load_fashion_mnist_dataset', 'FASHION_MNIST']

FASHION_MNIST_BASE_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
FASHION_MNIST_TRAIN_IMAGE_FILENAME = 'train-images-idx3-ubyte.gz'
FASHION_MNIST_TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte.gz'
FASHION_MNIST_TEST_IMAGE_FILENAME = 't10k-images-idx3-ubyte.gz'
FASHION_MNIST_TEST_LABEL_FILENAME = 't10k-labels-idx1-ubyte.gz'


def load_fashion_mnist_dataset(shape=(-1, 784), name='fashion_mnist', path='raw_data'):
    """
    Load the fashion mnist.

    Automatically download fashion-MNIST dataset and return the training, validation and test set with 50000, 10000 and 10000 fashion images respectively, `examples <http://marubon-ds.blogspot.co.uk/2017/09/fashion-mnist-exploring.html>`__.

    Parameters
    ----------
    shape : tuple
        The shape of digit images (the default is (-1, 784), alternatively (-1, 28, 28, 1)).
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/fashion_mnist/``.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test: tuple
        Return splitted training/validation/test set respectively.

    Examples
    --------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist_dataset(shape=(-1,784), path='datasets')
    >>> X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    path = os.path.join(path, name)

    # Download and read the training and test set images and labels.
    logging.info("Load or Download {0} > {1}".format(name.upper(), path))
    X_train = _load_mnist_images(name=FASHION_MNIST_TRAIN_IMAGE_FILENAME, url=FASHION_MNIST_BASE_URL, path=path,
                                 shape=shape)
    y_train = _load_mnist_labels(name=FASHION_MNIST_TRAIN_LABEL_FILENAME, url=FASHION_MNIST_BASE_URL, path=path)
    X_test = _load_mnist_images(name=FASHION_MNIST_TEST_IMAGE_FILENAME, url=FASHION_MNIST_BASE_URL, path=path,
                                shape=shape)
    y_test = _load_mnist_labels(name=FASHION_MNIST_TEST_LABEL_FILENAME, url=FASHION_MNIST_BASE_URL, path=path)

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


class FASHION_MNIST(Dataset):
    """
    Load the fashion mnist.

    Automatically download fashion-MNIST dataset and return the training, validation and test set with 50000, 10000 and 10000 fashion images respectively, `examples <http://marubon-ds.blogspot.co.uk/2017/09/fashion-mnist-exploring.html>`__.

    Parameters
    ----------
    train_or_test : str
        Must be either 'train' or 'test'. Choose the training or test dataset.
    shape : tuple
        The shape of digit images (the default is (-1, 784), alternatively (-1, 28, 28, 1)).
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/fashion_mnist/``.
    """

    def __init__(self, train_or_test, name='fashion_mnist', path='raw_data', shape=(-1, 784)):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)

        assert train_or_test in ['train', 'test']
        if train_or_test == 'train':
            self.images = _load_mnist_images(name=FASHION_MNIST_TRAIN_IMAGE_FILENAME, url=FASHION_MNIST_BASE_URL,
                                             path=path,
                                             shape=shape)
            self.labels = _load_mnist_labels(name=FASHION_MNIST_TRAIN_LABEL_FILENAME, url=FASHION_MNIST_BASE_URL,
                                             path=path)
        else:
            self.images = _load_mnist_images(name=FASHION_MNIST_TEST_IMAGE_FILENAME, url=FASHION_MNIST_BASE_URL,
                                             path=path,
                                             shape=shape)
            self.labels = _load_mnist_labels(name=FASHION_MNIST_TEST_LABEL_FILENAME, url=FASHION_MNIST_BASE_URL,
                                             path=path)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
