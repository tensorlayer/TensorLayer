import logging
import os
import pickle
import sys
import numpy as np

from ..base import Dataset
from ..utils import maybe_download_and_extract

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR10(Dataset):
    def __init__(self, train_or_test, path='data', name='cifar10'):
        self.path = os.path.join(path, name)

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

        # Download and read the training and test set images and labels.
        logging.info("Load or Download {0} > {1}".format(name.upper(), self.path))

        filename = 'cifar-10-python.tar.gz'
        maybe_download_and_extract(filename, path, CIFAR10_URL, extract=True)

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

        self.images = self.images.reshape((-1, 32, 32, 3))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
