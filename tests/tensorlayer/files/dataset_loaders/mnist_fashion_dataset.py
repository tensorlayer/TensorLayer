#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.files.utils import _load_mnist_dataset

__all__ = ['load_fashion_mnist_dataset']


def load_fashion_mnist_dataset(shape=(-1, 784), path='data'):
    """Load the fashion mnist.

    Automatically download fashion-MNIST dataset and return the training, validation and test set with 50000, 10000 and 10000 fashion images respectively, `examples <http://marubon-ds.blogspot.co.uk/2017/09/fashion-mnist-exploring.html>`__.

    Parameters
    ----------
    shape : tuple
        The shape of digit images (the default is (-1, 784), alternatively (-1, 28, 28, 1)).
    path : str
        The path that the data is downloaded to.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test: tuple
        Return splitted training/validation/test set respectively.

    Examples
    --------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_fashion_mnist_dataset(shape=(-1,784), path='datasets')
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_fashion_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    return _load_mnist_dataset(
        shape, path, name='fashion_mnist', url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    )
