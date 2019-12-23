import gzip
import logging
import os
import numpy as np

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['MNIST', 'load_mnist_dataset']

MNIST_BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
MNIST_TRAIN_IMAGE_FILENAME = 'train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABEL_FILENAME = 'train-labels-idx1-ubyte.gz'
MNIST_TEST_IMAGE_FILENAME = 't10k-images-idx3-ubyte.gz'
MNIST_TEST_LABEL_FILENAME = 't10k-labels-idx1-ubyte.gz'


def _load_mnist_images(name, url, path, shape):
    filepath = maybe_download_and_extract(name, path, url)

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


def _load_mnist_labels(name, url, path):
    filepath = maybe_download_and_extract(name, path, url)

    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


def load_mnist_dataset(shape=(-1, 784), name='mnist', path='raw_data'):
    """
    A generic function to load mnist-like dataset.

    Parameters:
    ----------
    shape : tuple
        The shape of digit images.
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/mnist/``.
    """
    path = os.path.join(path, name)

    # Download and read the training and test set images and labels.
    logging.info("Load or Download {0} > {1}".format(name.upper(), path))
    X_train = _load_mnist_images(name=MNIST_TRAIN_IMAGE_FILENAME, url=MNIST_BASE_URL, path=path, shape=shape)
    y_train = _load_mnist_labels(name=MNIST_TRAIN_LABEL_FILENAME, url=MNIST_BASE_URL, path=path)
    X_test = _load_mnist_images(name=MNIST_TEST_IMAGE_FILENAME, url=MNIST_BASE_URL, path=path, shape=shape)
    y_test = _load_mnist_labels(name=MNIST_TEST_LABEL_FILENAME, url=MNIST_BASE_URL, path=path)

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


class MNIST(Dataset):
    """
    Load MNIST dataset.

    Parameters:
    ----------
    train_or_test : str
        Must be either 'train' or 'test'. Choose the training or test dataset.
    shape : tuple
        The shape of digit images.
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/mnist/``.
    """
    def __init__(self, train_or_test, path='raw_data', name='mnist', shape=(-1, 784)):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)

        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        if train_or_test == 'train':
            self.images = _load_mnist_images(name=MNIST_TRAIN_IMAGE_FILENAME, url=MNIST_BASE_URL, path=path,
                                             shape=shape)
            self.labels = _load_mnist_labels(name=MNIST_TRAIN_LABEL_FILENAME, url=MNIST_BASE_URL, path=path)
        else:
            self.images = _load_mnist_images(name=MNIST_TEST_IMAGE_FILENAME, url=MNIST_BASE_URL, path=path,
                                             shape=shape)
            self.labels = _load_mnist_labels(name=MNIST_TEST_LABEL_FILENAME, url=MNIST_BASE_URL, path=path)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
