import gzip
import logging
import os
import numpy as np

from ..base import Dataset
from ..utils import maybe_download_and_extract

MNIST_TRAIN_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
MNIST_TRAIN_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
MNIST_TEST_IMAGE_URL = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
MNIST_TEST_LABEL_URL = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'


class MNIST(Dataset):
    def __init__(self, train_or_test, path='data', name='mnist'):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)

        assert train_or_test in ['train', 'test']
        if train_or_test == 'train':
            self.images = self.load_mnist_images(train_or_test=train_or_test)
            self.labels = self.load_mnist_labels(train_or_test=train_or_test)
        else:
            self.images = self.load_mnist_images(train_or_test=train_or_test)
            self.labels = self.load_mnist_labels(train_or_test=train_or_test)

    def load_mnist_images(self, train_or_test):
        if train_or_test == 'train':
            filepath = maybe_download_and_extract('train-images-idx3-ubyte.gz', self.path, MNIST_TRAIN_IMAGE_URL)
        else:
            filepath = maybe_download_and_extract('t10k-images-idx3-ubyte.gz', self.path, MNIST_TEST_IMAGE_URL)

        logging.info(filepath)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape((-1, 28, 28, 1))
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(self, train_or_test):
        if train_or_test == 'train':
            filepath = maybe_download_and_extract('train-labels-idx1-ubyte.gz', self.path, MNIST_TRAIN_LABEL_URL)
        else:
            filepath = maybe_download_and_extract('t10k-labels-idx1-ubyte.gz', self.path, MNIST_TEST_LABEL_URL)

        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
