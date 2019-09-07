import logging
import os
import pickle
import sys
import numpy as np

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['load_cifar10_dataset', 'CIFAR10']

CIFAR10_BASE_URL = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_FILENAME = 'cifar-10-python.tar.gz'


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


def load_cifar10_dataset(shape=(-1, 32, 32, 3), path='raw_data', name='cifar10', plotable=False):
    """
    Load CIFAR-10 dataset.

    It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with
    10000 images. The test batch contains exactly 1000 randomly-selected images from
    each class. The training batches contain the remaining images in random order,
    but some training batches may contain more images from one class than another.
    Between them, the training batches contain exactly 5000 images from each class.

    Parameters
    ----------
    shape : tuple
        The shape of digit images e.g. (-1, 3, 32, 32) and (-1, 32, 32, 3).
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/cifar10/``.
    plotable : boolean
        Whether to plot some image examples, False as default.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = load_cifar10_dataset(shape=(-1, 32, 32, 3))

    References
    ----------
    - `CIFAR website <https://www.cs.toronto.edu/~kriz/cifar.html>`__
    - `Data download link <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`__
    - `<https://teratail.com/questions/28932>`__

    """
    path = os.path.join(path, name)
    logging.info("Load or Download cifar10 > {}".format(path))

    # Download and uncompress file
    maybe_download_and_extract(CIFAR10_FILENAME, path, CIFAR10_BASE_URL, extract=True)

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

    if plotable:
        logging.info('\nCIFAR-10')
        import matplotlib.pyplot as plt
        fig = plt.figure(1)

        logging.info('Shape of a training image: X_train[0] %s' % X_train[0].shape)

        plt.ion()  # interactive mode
        count = 1
        for _ in range(10):  # each row
            for _ in range(10):  # each column
                _ = fig.add_subplot(10, 10, count)
                if shape == (-1, 3, 32, 32):
                    # plt.imshow(X_train[count-1], interpolation='nearest')
                    plt.imshow(np.transpose(X_train[count - 1], (1, 2, 0)), interpolation='nearest')
                    # plt.imshow(np.transpose(X_train[count-1], (2, 1, 0)), interpolation='nearest')
                elif shape == (-1, 32, 32, 3):
                    plt.imshow(X_train[count - 1], interpolation='nearest')
                    # plt.imshow(np.transpose(X_train[count-1], (1, 0, 2)), interpolation='nearest')
                else:
                    raise Exception("Do not support the given 'shape' to plot the image examples")
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                count = count + 1
        plt.draw()  # interactive mode
        plt.pause(3)  # interactive mode

        logging.info("X_train: %s" % X_train.shape)
        logging.info("y_train: %s" % y_train.shape)
        logging.info("X_test:  %s" % X_test.shape)
        logging.info("y_test:  %s" % y_test.shape)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test


class CIFAR10(Dataset):
    """
    Load CIFAR-10 dataset.

    It consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.

    Parameters
    ----------
    train_or_test : str
        Must be either 'train' or 'test'. Choose the training or test dataset.
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/cifar10/``.
    shape : tuple
        The shape of digit images e.g. (-1, 3, 32, 32) and (-1, 32, 32, 3).
    """
    def __init__(self, train_or_test, path='raw_data', name='cifar10', shape=(-1, 32, 32, 3)):
        self.path = os.path.join(path, name)

        # Download and read the training and test set images and labels.
        logging.info("Load or Download {0} > {1}".format(name.upper(), self.path))

        maybe_download_and_extract(CIFAR10_FILENAME, path, CIFAR10_BASE_URL, extract=True)

        assert train_or_test in ['train', 'test']
        if train_or_test == 'train':
            # Unpickle file and fill in data
            self.images = None
            self.labels = []
            for i in range(1, 6):
                data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', "data_batch_{}".format(i)))
                if i == 1:
                    self.images = data_dic['data']
                else:
                    self.images = np.vstack((self.images, data_dic['data']))
                self.labels += data_dic['labels']
        else:
            test_data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', "test_batch"))
            self.images = test_data_dic['data']
            self.labels = np.array(test_data_dic['labels'])

        self.images = np.reshape(self.images, shape)

    def __getitem__(self, index):
        img = np.array(self.images[index], dtype=np.float32)
        label = np.array(self.labels[index], dtype=np.int32)
        return img, label

    def __len__(self):
        return self.images.shape[0]
