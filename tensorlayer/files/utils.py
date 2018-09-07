#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
# import ast
import sys
import gzip
import math
import pickle
import progressbar
import re
import requests
import shutil
import tarfile
import time
import zipfile
import importlib
from tqdm import tqdm

from six.moves import cPickle
# from six.moves import zip

from lxml import etree
import xml.etree.ElementTree as ET

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# Fix error on OSX, as suggested by: https://stackoverflow.com/a/48374671
# See: https://docs.python.org/3/library/sys.html#sys.platform
if sys.platform.startswith('darwin'):
    import matplotlib
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer import nlp
from tensorlayer import utils
from tensorlayer import visualize

__all__ = [
    'assign_params',
    'del_file',
    'del_folder',
    'download_file_from_google_drive',
    'exists_or_mkdir',
    'file_exists',
    'folder_exists',
    'load_and_assign_npz',
    'load_and_assign_npz_dict',
    'load_ckpt',
    'load_cropped_svhn',
    'load_file_list',
    'load_folder_list',
    'load_npy_to_any',
    'load_npz',
    'maybe_download_and_extract',
    'natural_keys',
    'npz_to_W_pdf',
    'read_file',
    'save_any_to_npy',
    'save_ckpt',
    'save_npz',
    'save_npz_dict',
    #'save_graph',
    #'load_graph',
    #'save_graph_and_params',
    #'load_graph_and_params',
]


# Load dataset functions
def load_mnist_dataset(shape=(-1, 784), path='data'):
    """Load the original mnist.

    Automatically download MNIST dataset and return the training, validation and test set with 50000, 10000 and 10000 digit images respectively.

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
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784), path='datasets')
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    return _load_mnist_dataset(shape, path, name='mnist', url='http://yann.lecun.com/exdb/mnist/')


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


def load_cifar10_dataset(shape=(-1, 32, 32, 3), path='data', plotable=False):
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

    if plotable:
        logging.info('\nCIFAR-10')
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
                plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 不显示刻度(tick)
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


def load_cropped_svhn(path='data', include_extra=True):
    """Load Cropped SVHN.

    The Cropped Street View House Numbers (SVHN) Dataset contains 32x32x3 RGB images.
    Digit '1' has label 1, '9' has label 9 and '0' has label 0 (the original dataset uses 10 to represent '0'), see `ufldl website <http://ufldl.stanford.edu/housenumbers/>`__.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to.
    include_extra : boolean
        If True (default), add extra images to the training set.

    Returns
    -------
    X_train, y_train, X_test, y_test: tuple
        Return splitted training/test set respectively.

    Examples
    ---------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)
    >>> tl.vis.save_images(X_train[0:100], [10, 10], 'svhn.png')

    """
    start_time = time.time()

    path = os.path.join(path, 'cropped_svhn')
    logging.info("Load or Download Cropped SVHN > {} | include extra images: {}".format(path, include_extra))
    url = "http://ufldl.stanford.edu/housenumbers/"

    np_file = os.path.join(path, "train_32x32.npz")
    if file_exists(np_file) is False:
        filename = "train_32x32.mat"
        filepath = maybe_download_and_extract(filename, path, url)
        mat = sio.loadmat(filepath)
        X_train = mat['X'] / 255.0  # to [0, 1]
        X_train = np.transpose(X_train, (3, 0, 1, 2))
        y_train = np.squeeze(mat['y'], axis=1)
        y_train[y_train == 10] = 0  # replace 10 to 0
        np.savez(np_file, X=X_train, y=y_train)
        del_file(filepath)
    else:
        v = np.load(np_file)
        X_train = v['X']
        y_train = v['y']
    logging.info("  n_train: {}".format(len(y_train)))

    np_file = os.path.join(path, "test_32x32.npz")
    if file_exists(np_file) is False:
        filename = "test_32x32.mat"
        filepath = maybe_download_and_extract(filename, path, url)
        mat = sio.loadmat(filepath)
        X_test = mat['X'] / 255.0
        X_test = np.transpose(X_test, (3, 0, 1, 2))
        y_test = np.squeeze(mat['y'], axis=1)
        y_test[y_test == 10] = 0
        np.savez(np_file, X=X_test, y=y_test)
        del_file(filepath)
    else:
        v = np.load(np_file)
        X_test = v['X']
        y_test = v['y']
    logging.info("  n_test: {}".format(len(y_test)))

    if include_extra:
        logging.info("  getting extra 531131 images, please wait ...")
        np_file = os.path.join(path, "extra_32x32.npz")
        if file_exists(np_file) is False:
            logging.info("  the first time to load extra images will take long time to convert the file format ...")
            filename = "extra_32x32.mat"
            filepath = maybe_download_and_extract(filename, path, url)
            mat = sio.loadmat(filepath)
            X_extra = mat['X'] / 255.0
            X_extra = np.transpose(X_extra, (3, 0, 1, 2))
            y_extra = np.squeeze(mat['y'], axis=1)
            y_extra[y_extra == 10] = 0
            np.savez(np_file, X=X_extra, y=y_extra)
            del_file(filepath)
        else:
            v = np.load(np_file)
            X_extra = v['X']
            y_extra = v['y']
        # print(X_train.shape, X_extra.shape)
        logging.info("  adding n_extra {} to n_train {}".format(len(y_extra), len(y_train)))
        t = time.time()
        X_train = np.concatenate((X_train, X_extra), 0)
        y_train = np.concatenate((y_train, y_extra), 0)
        # X_train = np.append(X_train, X_extra, axis=0)
        # y_train = np.append(y_train, y_extra, axis=0)
        logging.info("  added n_extra {} to n_train {} took {}s".format(len(y_extra), len(y_train), time.time() - t))
    else:
        logging.info("  no extra images are included")
    logging.info("  image size: %s n_train: %d n_test: %d" % (str(X_train.shape[1:4]), len(y_train), len(y_test)))
    logging.info("  took: {}s".format(int(time.time() - start_time)))
    return X_train, y_train, X_test, y_test


def load_ptb_dataset(path='data'):
    """Load Penn TreeBank (PTB) dataset.

    It is used in many LANGUAGE MODELING papers,
    including "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques", "Recurrent Neural Network Regularization".
    It consists of 929k training words, 73k validation words, and 82k test
    words. It has 10k words in its vocabulary.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/ptb/``.

    Returns
    --------
    train_data, valid_data, test_data : list of int
        The training, validating and testing data in integer format.
    vocab_size : int
        The vocabulary size.

    Examples
    --------
    >>> train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()

    References
    ---------------
    - ``tensorflow.models.rnn.ptb import reader``
    - `Manual download <http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz>`__

    Notes
    ------
    - If you want to get the raw data, see the source code.

    """
    path = os.path.join(path, 'ptb')
    logging.info("Load or Download Penn TreeBank (PTB) dataset > {}".format(path))

    # Maybe dowload and uncompress tar, or load exsisting files
    filename = 'simple-examples.tgz'
    url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
    maybe_download_and_extract(filename, path, url, extract=True)

    data_path = os.path.join(path, 'simple-examples', 'data')
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = nlp.build_vocab(nlp.read_words(train_path))

    train_data = nlp.words_to_word_ids(nlp.read_words(train_path), word_to_id)
    valid_data = nlp.words_to_word_ids(nlp.read_words(valid_path), word_to_id)
    test_data = nlp.words_to_word_ids(nlp.read_words(test_path), word_to_id)
    vocab_size = len(word_to_id)

    # logging.info(nlp.read_words(train_path)) # ... 'according', 'to', 'mr.', '<unk>', '<eos>']
    # logging.info(train_data)                 # ...  214,         5,    23,    1,       2]
    # logging.info(word_to_id)                 # ... 'beyond': 1295, 'anti-nuclear': 9599, 'trouble': 1520, '<eos>': 2 ... }
    # logging.info(vocabulary)                 # 10000
    # exit()
    return train_data, valid_data, test_data, vocab_size


def load_matt_mahoney_text8_dataset(path='data'):
    """Load Matt Mahoney's dataset.

    Download a text file from Matt Mahoney's website
    if not present, and make sure it's the right size.
    Extract the first file enclosed in a zip file as a list of words.
    This dataset can be used for Word Embedding.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/mm_test8/``.

    Returns
    --------
    list of str
        The raw text data e.g. [.... 'their', 'families', 'who', 'were', 'expelled', 'from', 'jerusalem', ...]

    Examples
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> print('Data size', len(words))

    """
    path = os.path.join(path, 'mm_test8')
    logging.info("Load or Download matt_mahoney_text8 Dataset> {}".format(path))

    filename = 'text8.zip'
    url = 'http://mattmahoney.net/dc/'
    maybe_download_and_extract(filename, path, url, expected_bytes=31344016)

    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        word_list = f.read(f.namelist()[0]).split()
        for idx, _ in enumerate(word_list):
            word_list[idx] = word_list[idx].decode()
    return word_list


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

    X, labels = cPickle.load(f)
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


def load_nietzsche_dataset(path='data'):
    """Load Nietzsche dataset.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/nietzsche/``.

    Returns
    --------
    str
        The content.

    Examples
    --------
    >>> see tutorial_generate_text.py
    >>> words = tl.files.load_nietzsche_dataset()
    >>> words = basic_clean_str(words)
    >>> words = words.split()

    """
    logging.info("Load or Download nietzsche dataset > {}".format(path))
    path = os.path.join(path, 'nietzsche')

    filename = "nietzsche.txt"
    url = 'https://s3.amazonaws.com/text-datasets/'
    filepath = maybe_download_and_extract(filename, path, url)

    with open(filepath, "r") as f:
        words = f.read()
        return words


def load_wmt_en_fr_dataset(path='data'):
    """Load WMT'15 English-to-French translation dataset.

    It will download the data from the WMT'15 Website (10^9-French-English corpus), and the 2013 news test from the same site as development set.
    Returns the directories of training data and test data.

    Parameters
    ----------
    path : str
        The path that the data is downloaded to, defaults is ``data/wmt_en_fr/``.

    References
    ----------
    - Code modified from /tensorflow/models/rnn/translation/data_utils.py

    Notes
    -----
    Usually, it will take a long time to download this dataset.

    """
    path = os.path.join(path, 'wmt_en_fr')
    # URLs for WMT data.
    _WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/"
    _WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/"

    def gunzip_file(gz_path, new_path):
        """Unzips from gz_path into new_path."""
        logging.info("Unpacking %s to %s" % (gz_path, new_path))
        with gzip.open(gz_path, "rb") as gz_file:
            with open(new_path, "wb") as new_file:
                for line in gz_file:
                    new_file.write(line)

    def get_wmt_enfr_train_set(path):
        """Download the WMT en-fr training corpus to directory unless it's there."""
        filename = "training-giga-fren.tar"
        maybe_download_and_extract(filename, path, _WMT_ENFR_TRAIN_URL, extract=True)
        train_path = os.path.join(path, "giga-fren.release2.fixed")
        gunzip_file(train_path + ".fr.gz", train_path + ".fr")
        gunzip_file(train_path + ".en.gz", train_path + ".en")
        return train_path

    def get_wmt_enfr_dev_set(path):
        """Download the WMT en-fr training corpus to directory unless it's there."""
        filename = "dev-v2.tgz"
        dev_file = maybe_download_and_extract(filename, path, _WMT_ENFR_DEV_URL, extract=False)
        dev_name = "newstest2013"
        dev_path = os.path.join(path, "newstest2013")
        if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
            logging.info("Extracting tgz file %s" % dev_file)
            with tarfile.open(dev_file, "r:gz") as dev_tar:
                fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
                en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
                fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
                en_dev_file.name = dev_name + ".en"
                dev_tar.extract(fr_dev_file, path)
                dev_tar.extract(en_dev_file, path)
        return dev_path

    logging.info("Load or Download WMT English-to-French translation > {}".format(path))

    train_path = get_wmt_enfr_train_set(path)
    dev_path = get_wmt_enfr_dev_set(path)

    return train_path, dev_path


def load_flickr25k_dataset(tag='sky', path="data", n_threads=50, printable=False):
    """Load Flickr25K dataset.

    Returns a list of images by a given tag from Flick25k dataset,
    it will download Flickr25k from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`__
    at the first time you use it.

    Parameters
    ------------
    tag : str or None
        What images to return.
            - If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`__.
            - If you want to get all images, set to ``None``.

    path : str
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int
        The number of thread to read image.
    printable : boolean
        Whether to print infomation when reading images, default is ``False``.

    Examples
    -----------
    Get images with tag of sky

    >>> images = tl.files.load_flickr25k_dataset(tag='sky')

    Get all images

    >>> images = tl.files.load_flickr25k_dataset(tag=None, n_threads=100, printable=True)

    """
    path = os.path.join(path, 'flickr25k')

    filename = 'mirflickr25k.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr25k/'

    # download dataset
    if folder_exists(os.path.join(path, "mirflickr")) is False:
        logging.info("[*] Flickr25k is nonexistent in {}".format(path))
        maybe_download_and_extract(filename, path, url, extract=True)
        del_file(os.path.join(path, filename))

    # return images by the given tag.
    # 1. image path list
    folder_imgs = os.path.join(path, "mirflickr")
    path_imgs = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    path_imgs.sort(key=natural_keys)

    # 2. tag path list
    folder_tags = os.path.join(path, "mirflickr", "meta", "tags")
    path_tags = load_file_list(path=folder_tags, regx='\\.txt', printable=False)
    path_tags.sort(key=natural_keys)

    # 3. select images
    if tag is None:
        logging.info("[Flickr25k] reading all images")
    else:
        logging.info("[Flickr25k] reading images with tag: {}".format(tag))
    images_list = []
    for idx, _v in enumerate(path_tags):
        tags = read_file(os.path.join(folder_tags, path_tags[idx])).split('\n')
        # logging.info(idx+1, tags)
        if tag is None or tag in tags:
            images_list.append(path_imgs[idx])

    images = visualize.read_images(images_list, folder_imgs, n_threads=n_threads, printable=printable)
    return images


def load_flickr1M_dataset(tag='sky', size=10, path="data", n_threads=50, printable=False):
    """Load Flick1M dataset.

    Returns a list of images by a given tag from Flickr1M dataset,
    it will download Flickr1M from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`__
    at the first time you use it.

    Parameters
    ------------
    tag : str or None
        What images to return.
            - If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`__.
            - If you want to get all images, set to ``None``.

    size : int
        integer between 1 to 10. 1 means 100k images ... 5 means 500k images, 10 means all 1 million images. Default is 10.
    path : str
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int
        The number of thread to read image.
    printable : boolean
        Whether to print infomation when reading images, default is ``False``.

    Examples
    ----------
    Use 200k images

    >>> images = tl.files.load_flickr1M_dataset(tag='zebra', size=2)

    Use 1 Million images

    >>> images = tl.files.load_flickr1M_dataset(tag='zebra')

    """
    path = os.path.join(path, 'flickr1M')
    logging.info("[Flickr1M] using {}% of images = {}".format(size * 10, size * 100000))
    images_zip = [
        'images0.zip', 'images1.zip', 'images2.zip', 'images3.zip', 'images4.zip', 'images5.zip', 'images6.zip',
        'images7.zip', 'images8.zip', 'images9.zip'
    ]
    tag_zip = 'tags.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr1m/'

    # download dataset
    for image_zip in images_zip[0:size]:
        image_folder = image_zip.split(".")[0]
        # logging.info(path+"/"+image_folder)
        if folder_exists(os.path.join(path, image_folder)) is False:
            # logging.info(image_zip)
            logging.info("[Flickr1M] {} is missing in {}".format(image_folder, path))
            maybe_download_and_extract(image_zip, path, url, extract=True)
            del_file(os.path.join(path, image_zip))
            # os.system("mv {} {}".format(os.path.join(path, 'images'), os.path.join(path, image_folder)))
            shutil.move(os.path.join(path, 'images'), os.path.join(path, image_folder))
        else:
            logging.info("[Flickr1M] {} exists in {}".format(image_folder, path))

    # download tag
    if folder_exists(os.path.join(path, "tags")) is False:
        logging.info("[Flickr1M] tag files is nonexistent in {}".format(path))
        maybe_download_and_extract(tag_zip, path, url, extract=True)
        del_file(os.path.join(path, tag_zip))
    else:
        logging.info("[Flickr1M] tags exists in {}".format(path))

    # 1. image path list
    images_list = []
    images_folder_list = []
    for i in range(0, size):
        images_folder_list += load_folder_list(path=os.path.join(path, 'images%d' % i))
    images_folder_list.sort(key=lambda s: int(s.split('/')[-1]))  # folder/images/ddd

    for folder in images_folder_list[0:size * 10]:
        tmp = load_file_list(path=folder, regx='\\.jpg', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.jpg
        images_list.extend([os.path.join(folder, x) for x in tmp])

    # 2. tag path list
    tag_list = []
    tag_folder_list = load_folder_list(os.path.join(path, "tags"))

    # tag_folder_list.sort(key=lambda s: int(s.split("/")[-1]))  # folder/images/ddd
    tag_folder_list.sort(key=lambda s: int(os.path.basename(s)))

    for folder in tag_folder_list[0:size * 10]:
        tmp = load_file_list(path=folder, regx='\\.txt', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.txt
        tmp = [os.path.join(folder, s) for s in tmp]
        tag_list += tmp

    # 3. select images
    logging.info("[Flickr1M] searching tag: {}".format(tag))
    select_images_list = []
    for idx, _val in enumerate(tag_list):
        tags = read_file(tag_list[idx]).split('\n')
        if tag in tags:
            select_images_list.append(images_list[idx])

    logging.info("[Flickr1M] reading images with tag: {}".format(tag))
    images = visualize.read_images(select_images_list, '', n_threads=n_threads, printable=printable)
    return images


def load_cyclegan_dataset(filename='summer2winter_yosemite', path='data'):
    """Load images from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`__.

    Parameters
    ------------
    filename : str
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`__.
    path : str
        The path that the data is downloaded to, defaults is `data/cyclegan`

    Examples
    ---------
    >>> im_train_A, im_train_B, im_test_A, im_test_B = load_cyclegan_dataset(filename='summer2winter_yosemite')

    """
    path = os.path.join(path, 'cyclegan')
    url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

    if folder_exists(os.path.join(path, filename)) is False:
        logging.info("[*] {} is nonexistent in {}".format(filename, path))
        maybe_download_and_extract(filename + '.zip', path, url, extract=True)
        del_file(os.path.join(path, filename + '.zip'))

    def load_image_from_folder(path):
        path_imgs = load_file_list(path=path, regx='\\.jpg', printable=False)
        return visualize.read_images(path_imgs, path=path, n_threads=10, printable=False)

    im_train_A = load_image_from_folder(os.path.join(path, filename, "trainA"))
    im_train_B = load_image_from_folder(os.path.join(path, filename, "trainB"))
    im_test_A = load_image_from_folder(os.path.join(path, filename, "testA"))
    im_test_B = load_image_from_folder(os.path.join(path, filename, "testB"))

    def if_2d_to_3d(images):  # [h, w] --> [h, w, 3]
        for i, _v in enumerate(images):
            if len(images[i].shape) == 2:
                images[i] = images[i][:, :, np.newaxis]
                images[i] = np.tile(images[i], (1, 1, 3))
        return images

    im_train_A = if_2d_to_3d(im_train_A)
    im_train_B = if_2d_to_3d(im_train_B)
    im_test_A = if_2d_to_3d(im_test_A)
    im_test_B = if_2d_to_3d(im_test_B)

    return im_train_A, im_train_B, im_test_A, im_test_B


def download_file_from_google_drive(ID, destination):
    """Download file from Google Drive.

    See ``tl.files.load_celebA_dataset`` for example.

    Parameters
    --------------
    ID : str
        The driver ID.
    destination : str
        The destination for save file.

    """

    def save_response_content(response, destination, chunk_size=32 * 1024):
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_scale=True,
                              desc=destination):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': ID}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': ID, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def load_celebA_dataset(path='data'):
    """Load CelebA dataset

    Return a list of image path.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/celebA/``.

    """
    data_dir = 'celebA'
    filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(path, filename)
    image_path = os.path.join(path, data_dir)
    if os.path.exists(image_path):
        logging.info('[*] {} already exists'.format(save_path))
    else:
        exists_or_mkdir(path)
        download_file_from_google_drive(drive_id, save_path)
        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(path)
        os.remove(save_path)
        os.rename(os.path.join(path, zip_dir), image_path)

    data_files = load_file_list(path=image_path, regx='\\.jpg', printable=False)
    for i, _v in enumerate(data_files):
        data_files[i] = os.path.join(image_path, data_files[i])
    return data_files


def load_voc_dataset(path='data', dataset='2012', contain_classes_in_person=False):
    """Pascal VOC 2007/2012 Dataset.

    It has 20 objects:
    aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
    and additional 3 classes : head, hand, foot for person.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/VOC``.
    dataset : str
        The VOC dataset version, `2012`, `2007`, `2007test` or `2012test`. We usually train model on `2007+2012` and test it on `2007test`.
    contain_classes_in_person : boolean
        Whether include head, hand and foot annotation, default is False.

    Returns
    ---------
    imgs_file_list : list of str
        Full paths of all images.
    imgs_semseg_file_list : list of str
        Full paths of all maps for semantic segmentation. Note that not all images have this map!
    imgs_insseg_file_list : list of str
        Full paths of all maps for instance segmentation. Note that not all images have this map!
    imgs_ann_file_list : list of str
        Full paths of all annotations for bounding box and object class, all images have this annotations.
    classes : list of str
        Classes in order.
    classes_in_person : list of str
        Classes in person.
    classes_dict : dictionary
        Class label to integer.
    n_objs_list : list of int
        Number of objects in all images in ``imgs_file_list`` in order.
    objs_info_list : list of str
        Darknet format for the annotation of all images in ``imgs_file_list`` in order. ``[class_id x_centre y_centre width height]`` in ratio format.
    objs_info_dicts : dictionary
        The annotation of all images in ``imgs_file_list``, ``{imgs_file_list : dictionary for annotation}``,
        format from `TensorFlow/Models/object-detection <https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py>`__.

    Examples
    ----------
    >>> imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list,
    >>>     classes, classes_in_person, classes_dict,
    >>>     n_objs_list, objs_info_list, objs_info_dicts = tl.files.load_voc_dataset(dataset="2012", contain_classes_in_person=False)
    >>> idx = 26
    >>> print(classes)
    ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    >>> print(classes_dict)
    {'sheep': 16, 'horse': 12, 'bicycle': 1, 'bottle': 4, 'cow': 9, 'sofa': 17, 'car': 6, 'dog': 11, 'cat': 7, 'person': 14, 'train': 18, 'diningtable': 10, 'aeroplane': 0, 'bus': 5, 'pottedplant': 15, 'tvmonitor': 19, 'chair': 8, 'bird': 2, 'boat': 3, 'motorbike': 13}
    >>> print(imgs_file_list[idx])
    data/VOC/VOC2012/JPEGImages/2007_000423.jpg
    >>> print(n_objs_list[idx])
    2
    >>> print(imgs_ann_file_list[idx])
    data/VOC/VOC2012/Annotations/2007_000423.xml
    >>> print(objs_info_list[idx])
    14 0.173 0.461333333333 0.142 0.496
    14 0.828 0.542666666667 0.188 0.594666666667
    >>> ann = tl.prepro.parse_darknet_ann_str_to_list(objs_info_list[idx])
    >>> print(ann)
    [[14, 0.173, 0.461333333333, 0.142, 0.496], [14, 0.828, 0.542666666667, 0.188, 0.594666666667]]
    >>> c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    >>> print(c, b)
    [14, 14] [[0.173, 0.461333333333, 0.142, 0.496], [0.828, 0.542666666667, 0.188, 0.594666666667]]

    References
    -------------
    - `Pascal VOC2012 Website <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`__.
    - `Pascal VOC2007 Website <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>`__.

    """
    path = os.path.join(path, 'VOC')

    def _recursive_parse_xml_to_dict(xml):
        """Recursively parses XML contents to python dict.

        We assume that `object` tags are the only ones that can appear
        multiple times at the same level of a tree.

        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.

        """
        if not xml:
            # if xml is not None:
            return {xml.tag: xml.text}
        result = {}
        for child in xml:
            child_result = _recursive_parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    if dataset == "2012":
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
        tar_filename = "VOCtrainval_11-May-2012.tar"
        extracted_filename = "VOC2012"  # "VOCdevkit/VOC2012"
        logging.info("    [============= VOC 2012 =============]")
    elif dataset == "2012test":
        extracted_filename = "VOC2012test"  # "VOCdevkit/VOC2012"
        logging.info("    [============= VOC 2012 Test Set =============]")
        logging.info(
            "    \nAuthor: 2012test only have person annotation, so 2007test is highly recommended for testing !\n"
        )
        time.sleep(3)
        if os.path.isdir(os.path.join(path, extracted_filename)) is False:
            logging.info("For VOC 2012 Test data - online registration required")
            logging.info(
                " Please download VOC2012test.tar from:  \n register: http://host.robots.ox.ac.uk:8080 \n voc2012 : http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/ \ndownload: http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar"
            )
            logging.info(" unzip VOC2012test.tar,rename the folder to VOC2012test and put it into %s" % path)
            exit()
        # # http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
        # url = "http://host.robots.ox.ac.uk:8080/eval/downloads/"
        # tar_filename = "VOC2012test.tar"
    elif dataset == "2007":
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        tar_filename = "VOCtrainval_06-Nov-2007.tar"
        extracted_filename = "VOC2007"
        logging.info("    [============= VOC 2007 =============]")
    elif dataset == "2007test":
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#testdata
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        tar_filename = "VOCtest_06-Nov-2007.tar"
        extracted_filename = "VOC2007test"
        logging.info("    [============= VOC 2007 Test Set =============]")
    else:
        raise Exception("Please set the dataset aug to 2012, 2012test or 2007.")

    # download dataset
    if dataset != "2012test":
        _platform = sys.platform
        if folder_exists(os.path.join(path, extracted_filename)) is False:
            logging.info("[VOC] {} is nonexistent in {}".format(extracted_filename, path))
            maybe_download_and_extract(tar_filename, path, url, extract=True)
            del_file(os.path.join(path, tar_filename))
            if dataset == "2012":
                if _platform == "win32":
                    os.system("mv {}\VOCdevkit\VOC2012 {}\VOC2012".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2012 {}/VOC2012".format(path, path))
            elif dataset == "2007":
                if _platform == "win32":
                    os.system("mv {}\VOCdevkit\VOC2007 {}\VOC2007".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2007 {}/VOC2007".format(path, path))
            elif dataset == "2007test":
                if _platform == "win32":
                    os.system("mv {}\VOCdevkit\VOC2007 {}\VOC2007test".format(path, path))
                else:
                    os.system("mv {}/VOCdevkit/VOC2007 {}/VOC2007test".format(path, path))
            del_folder(os.path.join(path, 'VOCdevkit'))
    # object classes(labels)  NOTE: YOU CAN CUSTOMIZE THIS LIST
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    if contain_classes_in_person:
        classes_in_person = ["head", "hand", "foot"]
    else:
        classes_in_person = []

    classes += classes_in_person  # use extra 3 classes for person

    classes_dict = utils.list_string_to_dict(classes)
    logging.info("[VOC] object classes {}".format(classes_dict))

    # 1. image path list
    # folder_imgs = path+"/"+extracted_filename+"/JPEGImages/"
    folder_imgs = os.path.join(path, extracted_filename, "JPEGImages")
    imgs_file_list = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    logging.info("[VOC] {} images found".format(len(imgs_file_list)))

    imgs_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                       )  # 2007_000027.jpg --> 2007000027

    imgs_file_list = [os.path.join(folder_imgs, s) for s in imgs_file_list]
    # logging.info('IM',imgs_file_list[0::3333], imgs_file_list[-1])
    if dataset != "2012test":
        # ======== 2. semantic segmentation maps path list
        # folder_semseg = path+"/"+extracted_filename+"/SegmentationClass/"
        folder_semseg = os.path.join(path, extracted_filename, "SegmentationClass")
        imgs_semseg_file_list = load_file_list(path=folder_semseg, regx='\\.png', printable=False)
        logging.info("[VOC] {} maps for semantic segmentation found".format(len(imgs_semseg_file_list)))
        imgs_semseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                                  )  # 2007_000032.png --> 2007000032
        imgs_semseg_file_list = [os.path.join(folder_semseg, s) for s in imgs_semseg_file_list]
        # logging.info('Semantic Seg IM',imgs_semseg_file_list[0::333], imgs_semseg_file_list[-1])
        # ======== 3. instance segmentation maps path list
        # folder_insseg = path+"/"+extracted_filename+"/SegmentationObject/"
        folder_insseg = os.path.join(path, extracted_filename, "SegmentationObject")
        imgs_insseg_file_list = load_file_list(path=folder_insseg, regx='\\.png', printable=False)
        logging.info("[VOC] {} maps for instance segmentation found".format(len(imgs_semseg_file_list)))
        imgs_insseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                                  )  # 2007_000032.png --> 2007000032
        imgs_insseg_file_list = [os.path.join(folder_insseg, s) for s in imgs_insseg_file_list]
        # logging.info('Instance Seg IM',imgs_insseg_file_list[0::333], imgs_insseg_file_list[-1])
    else:
        imgs_semseg_file_list = []
        imgs_insseg_file_list = []
    # 4. annotations for bounding box and object class
    # folder_ann = path+"/"+extracted_filename+"/Annotations/"
    folder_ann = os.path.join(path, extracted_filename, "Annotations")
    imgs_ann_file_list = load_file_list(path=folder_ann, regx='\\.xml', printable=False)
    logging.info(
        "[VOC] {} XML annotation files for bounding box and object class found".format(len(imgs_ann_file_list))
    )
    imgs_ann_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2])
                           )  # 2007_000027.xml --> 2007000027
    imgs_ann_file_list = [os.path.join(folder_ann, s) for s in imgs_ann_file_list]
    # logging.info('ANN',imgs_ann_file_list[0::3333], imgs_ann_file_list[-1])

    if dataset == "2012test":  # remove unused images in JPEG folder
        imgs_file_list_new = []
        for ann in imgs_ann_file_list:
            ann = os.path.split(ann)[-1].split('.')[0]
            for im in imgs_file_list:
                if ann in im:
                    imgs_file_list_new.append(im)
                    break
        imgs_file_list = imgs_file_list_new
        logging.info("[VOC] keep %d images" % len(imgs_file_list_new))

    # parse XML annotations
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(file_name):
        """Given VOC2012 XML Annotations, returns number of objects and info."""
        in_file = open(file_name)
        out_file = ""
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        n_objs = 0

        for obj in root.iter('object'):
            if dataset != "2012test":
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
            else:
                cls = obj.find('name').text
                if cls not in classes:
                    continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )
            bb = convert((w, h), b)

            out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            n_objs += 1
            if cls in "person":
                for part in obj.iter('part'):
                    cls = part.find('name').text
                    if cls not in classes_in_person:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = part.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
                    )
                    bb = convert((w, h), b)
                    # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                    n_objs += 1
        in_file.close()
        return n_objs, out_file

    logging.info("[VOC] Parsing xml annotations files")
    n_objs_list = []
    objs_info_list = []  # Darknet Format list of string
    objs_info_dicts = {}
    for idx, ann_file in enumerate(imgs_ann_file_list):
        n_objs, objs_info = convert_annotation(ann_file)
        n_objs_list.append(n_objs)
        objs_info_list.append(objs_info)
        with tf.gfile.GFile(ann_file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = _recursive_parse_xml_to_dict(xml)['annotation']
        objs_info_dicts.update({imgs_file_list[idx]: data})

    return imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list, classes, classes_in_person, classes_dict, n_objs_list, objs_info_list, objs_info_dicts


def load_mpii_pose_dataset(path='data', is_16_pos_only=False):
    """Load MPII Human Pose Dataset.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to.
    is_16_pos_only : boolean
        If True, only return the peoples contain 16 pose keypoints. (Usually be used for single person pose estimation)

    Returns
    ----------
    img_train_list : list of str
        The image directories of training data.
    ann_train_list : list of dict
        The annotations of training data.
    img_test_list : list of str
        The image directories of testing data.
    ann_test_list : list of dict
        The annotations of testing data.

    Examples
    --------
    >>> import pprint
    >>> import tensorlayer as tl
    >>> img_train_list, ann_train_list, img_test_list, ann_test_list = tl.files.load_mpii_pose_dataset()
    >>> image = tl.vis.read_image(img_train_list[0])
    >>> tl.vis.draw_mpii_pose_to_image(image, ann_train_list[0], 'image.png')
    >>> pprint.pprint(ann_train_list[0])

    References
    -----------
    - `MPII Human Pose Dataset. CVPR 14 <http://human-pose.mpi-inf.mpg.de>`__
    - `MPII Human Pose Models. CVPR 16 <http://pose.mpi-inf.mpg.de>`__
    - `MPII Human Shape, Poselet Conditioned Pictorial Structures and etc <http://pose.mpi-inf.mpg.de/#related>`__
    - `MPII Keyponts and ID <http://human-pose.mpi-inf.mpg.de/#download>`__
    """
    path = os.path.join(path, 'mpii_human_pose')
    logging.info("Load or Download MPII Human Pose > {}".format(path))

    # annotation
    url = "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
    tar_filename = "mpii_human_pose_v1_u12_2.zip"
    extracted_filename = "mpii_human_pose_v1_u12_2"
    if folder_exists(os.path.join(path, extracted_filename)) is False:
        logging.info("[MPII] (annotation) {} is nonexistent in {}".format(extracted_filename, path))
        maybe_download_and_extract(tar_filename, path, url, extract=True)
        del_file(os.path.join(path, tar_filename))

    # images
    url = "http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/"
    tar_filename = "mpii_human_pose_v1.tar.gz"
    extracted_filename2 = "images"
    if folder_exists(os.path.join(path, extracted_filename2)) is False:
        logging.info("[MPII] (images) {} is nonexistent in {}".format(extracted_filename, path))
        maybe_download_and_extract(tar_filename, path, url, extract=True)
        del_file(os.path.join(path, tar_filename))

    # parse annotation, format see http://human-pose.mpi-inf.mpg.de/#download
    logging.info("reading annotations from mat file ...")
    # mat = sio.loadmat(os.path.join(path, extracted_filename, "mpii_human_pose_v1_u12_1.mat"))

    # def fix_wrong_joints(joint):    # https://github.com/mitmul/deeppose/blob/master/datasets/mpii_dataset.py
    #     if '12' in joint and '13' in joint and '2' in joint and '3' in joint:
    #         if ((joint['12'][0] < joint['13'][0]) and
    #                 (joint['3'][0] < joint['2'][0])):
    #             joint['2'], joint['3'] = joint['3'], joint['2']
    #         if ((joint['12'][0] > joint['13'][0]) and
    #                 (joint['3'][0] > joint['2'][0])):
    #             joint['2'], joint['3'] = joint['3'], joint['2']
    #     return joint

    ann_train_list = []
    ann_test_list = []
    img_train_list = []
    img_test_list = []

    def save_joints():
        # joint_data_fn = os.path.join(path, 'data.json')
        # fp = open(joint_data_fn, 'w')
        mat = sio.loadmat(os.path.join(path, extracted_filename, "mpii_human_pose_v1_u12_1.mat"))

        for _, (anno, train_flag) in enumerate(  # all images
                zip(mat['RELEASE']['annolist'][0, 0][0], mat['RELEASE']['img_train'][0, 0][0])):

            img_fn = anno['image']['name'][0, 0][0]
            train_flag = int(train_flag)

            # print(i, img_fn, train_flag) # DEBUG print all images

            if train_flag:
                img_train_list.append(img_fn)
                ann_train_list.append([])
            else:
                img_test_list.append(img_fn)
                ann_test_list.append([])

            head_rect = []
            if 'x1' in str(anno['annorect'].dtype):
                head_rect = zip(
                    [x1[0, 0] for x1 in anno['annorect']['x1'][0]], [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                    [x2[0, 0] for x2 in anno['annorect']['x2'][0]], [y2[0, 0] for y2 in anno['annorect']['y2'][0]]
                )
            else:
                head_rect = []  # TODO

            if 'annopoints' in str(anno['annorect'].dtype):
                annopoints = anno['annorect']['annopoints'][0]
                head_x1s = anno['annorect']['x1'][0]
                head_y1s = anno['annorect']['y1'][0]
                head_x2s = anno['annorect']['x2'][0]
                head_y2s = anno['annorect']['y2'][0]

                for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(annopoints, head_x1s, head_y1s, head_x2s,
                                                                         head_y2s):
                    # if annopoint != []:
                    # if len(annopoint) != 0:
                    if annopoint.size:
                        head_rect = [
                            float(head_x1[0, 0]),
                            float(head_y1[0, 0]),
                            float(head_x2[0, 0]),
                            float(head_y2[0, 0])
                        ]

                        # joint coordinates
                        annopoint = annopoint['point'][0, 0]
                        j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                        x = [x[0, 0] for x in annopoint['x'][0]]
                        y = [y[0, 0] for y in annopoint['y'][0]]
                        joint_pos = {}
                        for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                            joint_pos[int(_j_id)] = [float(_x), float(_y)]
                        # joint_pos = fix_wrong_joints(joint_pos)

                        # visibility list
                        if 'is_visible' in str(annopoint.dtype):
                            vis = [v[0] if v.size > 0 else [0] for v in annopoint['is_visible'][0]]
                            vis = dict([(k, int(v[0])) if len(v) > 0 else v for k, v in zip(j_id, vis)])
                        else:
                            vis = None

                        # if len(joint_pos) == 16:
                        if ((is_16_pos_only ==True) and (len(joint_pos) == 16)) or (is_16_pos_only == False):
                            # only use image with 16 key points / or use all
                            data = {
                                'filename': img_fn,
                                'train': train_flag,
                                'head_rect': head_rect,
                                'is_visible': vis,
                                'joint_pos': joint_pos
                            }
                            # print(json.dumps(data), file=fp)  # py3
                            if train_flag:
                                ann_train_list[-1].append(data)
                            else:
                                ann_test_list[-1].append(data)

    # def write_line(datum, fp):
    #     joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    #     joints = np.array([j for i, j in joints]).flatten()
    #
    #     out = [datum['filename']]
    #     out.extend(joints)
    #     out = [str(o) for o in out]
    #     out = ','.join(out)
    #
    #     print(out, file=fp)

    # def split_train_test():
    #     # fp_test = open('data/mpii/test_joints.csv', 'w')
    #     fp_test = open(os.path.join(path, 'test_joints.csv'), 'w')
    #     # fp_train = open('data/mpii/train_joints.csv', 'w')
    #     fp_train = open(os.path.join(path, 'train_joints.csv'), 'w')
    #     # all_data = open('data/mpii/data.json').readlines()
    #     all_data = open(os.path.join(path, 'data.json')).readlines()
    #     N = len(all_data)
    #     N_test = int(N * 0.1)
    #     N_train = N - N_test
    #
    #     print('N:{}'.format(N))
    #     print('N_train:{}'.format(N_train))
    #     print('N_test:{}'.format(N_test))
    #
    #     np.random.seed(1701)
    #     perm = np.random.permutation(N)
    #     test_indices = perm[:N_test]
    #     train_indices = perm[N_test:]
    #
    #     print('train_indices:{}'.format(len(train_indices)))
    #     print('test_indices:{}'.format(len(test_indices)))
    #
    #     for i in train_indices:
    #         datum = json.loads(all_data[i].strip())
    #         write_line(datum, fp_train)
    #
    #     for i in test_indices:
    #         datum = json.loads(all_data[i].strip())
    #         write_line(datum, fp_test)

    save_joints()
    # split_train_test()  #

    # read images dir
    logging.info("reading images list ...")
    img_dir = os.path.join(path, extracted_filename2)
    _img_list = load_file_list(path=os.path.join(path, extracted_filename2), regx='\\.jpg', printable=False)
    # ann_list = json.load(open(os.path.join(path, 'data.json')))
    for i, im in enumerate(img_train_list):
        if im not in _img_list:
            print('missing training image {} in {} (remove from img(ann)_train_list)'.format(im, img_dir))
            # img_train_list.remove(im)
            del img_train_list[i]
            del ann_train_list[i]
    for i, im in enumerate(img_test_list):
        if im not in _img_list:
            print('missing testing image {} in {} (remove from img(ann)_test_list)'.format(im, img_dir))
            # img_test_list.remove(im)
            del img_train_list[i]
            del ann_train_list[i]

    # check annotation and images
    n_train_images = len(img_train_list)
    n_test_images = len(img_test_list)
    n_images = n_train_images + n_test_images
    logging.info("n_images: {} n_train_images: {} n_test_images: {}".format(n_images, n_train_images, n_test_images))
    n_train_ann = len(ann_train_list)
    n_test_ann = len(ann_test_list)
    n_ann = n_train_ann + n_test_ann
    logging.info("n_ann: {} n_train_ann: {} n_test_ann: {}".format(n_ann, n_train_ann, n_test_ann))
    n_train_people = len(sum(ann_train_list, []))
    n_test_people = len(sum(ann_test_list, []))
    n_people = n_train_people + n_test_people
    logging.info("n_people: {} n_train_people: {} n_test_people: {}".format(n_people, n_train_people, n_test_people))
    # add path to all image file name
    for i, value in enumerate(img_train_list):
        img_train_list[i] = os.path.join(img_dir, value)
    for i, value in enumerate(img_test_list):
        img_test_list[i] = os.path.join(img_dir, value)
    return img_train_list, ann_train_list, img_test_list, ann_test_list


def save_npz(save_list=None, name='model.npz', sess=None):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.

    Parameters
    ----------
    save_list : list of tensor
        A list of parameters (tensor) to be saved.
    name : str
        The name of the `.npz` file.
    sess : None or Session
        Session may be required in some case.

    Examples
    --------
    Save model to npz

    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)

    Load model from npz (Method 1)

    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)

    Load model from npz (Method 2)

    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    Notes
    -----
    If you got session issues, you can change the value.eval() to value.eval(session=sess)

    References
    ----------
    `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`__

    """
    logging.info("[*] Saving TL params into %s" % name)
    if save_list is None:
        save_list = []

    save_list_var = []
    if sess:
        save_list_var = sess.run(save_list)
    else:
        try:
            save_list_var.extend([v.eval() for v in save_list])
        except Exception:
            logging.info(
                " Fail to save model, Hint: pass the session into this function, tl.files.save_npz(network.all_params, name='model.npz', sess=sess)"
            )
    np.savez(name, params=save_list_var)
    save_list_var = None
    del save_list_var
    logging.info("[*] Saved")


def load_npz(path='', name='model.npz'):
    """Load the parameters of a Model saved by tl.files.save_npz().

    Parameters
    ----------
    path : str
        Folder path to `.npz` file.
    name : str
        The name of the `.npz` file.

    Returns
    --------
    list of array
        A list of parameters in order.

    Examples
    --------
    - See ``tl.files.save_npz``

    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`__

    """
    d = np.load(os.path.join(path, name))
    return d['params']


def assign_params(sess, params, network):
    """Assign the given parameters to the TensorLayer network.

    Parameters
    ----------
    sess : Session
        TensorFlow Session.
    params : list of array
        A list of parameters (array) in order.
    network : :class:`Layer`
        The network to be assigned.

    Returns
    --------
    list of operations
        A list of tf ops in order that assign params. Support sess.run(ops) manually.

    Examples
    --------
    - See ``tl.files.save_npz``

    References
    ----------
    - `Assign value to a TensorFlow variable <http://stackoverflow.com/questions/34220532/how-to-assign-value-to-a-tensorflow-variable>`__

    """
    ops = []
    for idx, param in enumerate(params):
        ops.append(network.all_params[idx].assign(param))
    if sess is not None:
        sess.run(ops)
    return ops


def load_and_assign_npz(sess=None, name=None, network=None):
    """Load model from npz and assign to a network.

    Parameters
    -------------
    sess : Session
        TensorFlow Session.
    name : str
        The name of the `.npz` file.
    network : :class:`Layer`
        The network to be assigned.

    Returns
    --------
    False or network
        Returns False, if the model is not exist.

    Examples
    --------
    - See ``tl.files.save_npz``

    """
    if network is None:
        raise ValueError("network is None.")
    if sess is None:
        raise ValueError("session is None.")
    if not os.path.exists(name):
        logging.error("file {} doesn't exist.".format(name))
        return
    else:
        params = load_npz(name=name)
        assign_params(sess, params, network)
        logging.info("[*] Load {} SUCCESS!".format(name))
        return network


def save_npz_dict(save_list=None, name='model.npz', sess=None):
    """Input parameters and the file name, save parameters as a dictionary into .npz file.

    Use ``tl.files.load_and_assign_npz_dict()`` to restore.

    Parameters
    ----------
    save_list : list of parameters
        A list of parameters (tensor) to be saved.
    name : str
        The name of the `.npz` file.
    sess : Session
        TensorFlow Session.

    """
    if sess is None:
        raise ValueError("session is None.")
    if save_list is None:
        save_list = []

    save_list_names = [tensor.name for tensor in save_list]
    save_list_var = sess.run(save_list)
    save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
    np.savez(name, **save_var_dict)
    save_list_var = None
    save_var_dict = None
    del save_list_var
    del save_var_dict
    logging.info("[*] Model saved in npz_dict %s" % name)


def load_and_assign_npz_dict(name='model.npz', sess=None):
    """Restore the parameters saved by ``tl.files.save_npz_dict()``.

    Parameters
    ----------
    name : str
        The name of the `.npz` file.
    sess : Session
        TensorFlow Session.

    """
    if sess is None:
        raise ValueError("session is None.")

    if not os.path.exists(name):
        logging.error("file {} doesn't exist.".format(name))
        return

    params = np.load(name)
    if len(params.keys()) != len(set(params.keys())):
        raise Exception("Duplication in model npz_dict %s" % name)
    ops = list()
    for key in params.keys():
        try:
            # tensor = tf.get_default_graph().get_tensor_by_name(key)
            # varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=key)
            varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=key)
            if len(varlist) > 1:
                raise Exception("[!] Multiple candidate variables to be assigned for name %s" % key)
            elif len(varlist) == 0:
                raise KeyError
            else:
                ops.append(varlist[0].assign(params[key]))
                logging.info("[*] params restored: %s" % key)
        except KeyError:
            logging.info("[!] Warning: Tensor named %s not found in network." % key)

    sess.run(ops)
    logging.info("[*] Model restored from npz_dict %s" % name)


def save_ckpt(
        sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=None, global_step=None, printable=False
):
    """Save parameters into `ckpt` file.

    Parameters
    ------------
    sess : Session
        TensorFlow Session.
    mode_name : str
        The name of the model, default is ``model.ckpt``.
    save_dir : str
        The path / file directory to the `ckpt`, default is ``checkpoint``.
    var_list : list of tensor
        The parameters / variables (tensor) to be saved. If empty, save all global variables (default).
    global_step : int or None
        Step number.
    printable : boolean
        Whether to print all parameters information.

    See Also
    --------
    load_ckpt

    """
    if sess is None:
        raise ValueError("session is None.")
    if var_list is None:
        var_list = []

    ckpt_file = os.path.join(save_dir, mode_name)
    if var_list == []:
        var_list = tf.global_variables()

    logging.info("[*] save %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            logging.info("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    saver = tf.train.Saver(var_list)
    saver.save(sess, ckpt_file, global_step=global_step)


def load_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=None, is_latest=True, printable=False):
    """Load parameters from `ckpt` file.

    Parameters
    ------------
    sess : Session
        TensorFlow Session.
    mode_name : str
        The name of the model, default is ``model.ckpt``.
    save_dir : str
        The path / file directory to the `ckpt`, default is ``checkpoint``.
    var_list : list of tensor
        The parameters / variables (tensor) to be saved. If empty, save all global variables (default).
    is_latest : boolean
        Whether to load the latest `ckpt`, if False, load the `ckpt` with the name of ```mode_name``.
    printable : boolean
        Whether to print all parameters information.

    Examples
    ----------
    - Save all global parameters.

    >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir='model', printable=True)

    - Save specific parameters.

    >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', printable=True)

    - Load latest ckpt.

    >>> tl.files.load_ckpt(sess=sess, var_list=net.all_params, save_dir='model', printable=True)

    - Load specific ckpt.

    >>> tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', is_latest=False, printable=True)

    """
    if sess is None:
        raise ValueError("session is None.")
    if var_list is None:
        var_list = []

    if is_latest:
        ckpt_file = tf.train.latest_checkpoint(save_dir)
    else:
        ckpt_file = os.path.join(save_dir, mode_name)

    if not var_list:
        var_list = tf.global_variables()

    logging.info("[*] load %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            logging.info("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    try:
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)
    except Exception as e:
        logging.info(e)
        logging.info("[*] load ckpt fail ...")


'''
def save_graph(network=None, name='graph.pkl'):
    """Save the architecture of TL model into a pickle file. No parameters be saved.

    Parameters
    -----------
    network : TensorLayer layer
        The network to save.
    name : str
        The name of graph file.

    Examples
    --------
    Save the architecture
    >>> tl.files.save_graph(net_test, 'graph.pkl')

    Load the architecture in another script (no parameters restore)
    >>> net = tl.files.load_graph('graph.pkl')
    """
    logging.info("[*] Saving TL graph into {}".format(name))
    graphs = network.all_graphs
    with open(name, 'wb') as file:
        # pickle.dumps(graphs, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(graphs, file, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("[*] Saved graph")


def _graph2net(graphs):
    """Inputs graphs, returns network."""
    input_list = list()
    layer_dict = dict()
    # loop every layers
    for graph in graphs:
        # get current layer class
        name, layer_kwargs = graph
        layer_kwargs = dict(
            layer_kwargs
        )  # when InputLayer is used for twice, if we "pop" elements, the second time to use it will have error.

        layer_class = layer_kwargs.pop('class')  # class of current layer
        prev_layer = layer_kwargs.pop(
            'prev_layer'
        )  # name of previous layer : str =one layer   list of str = multiple layers

        # convert function dictionary into real function
        for key in layer_kwargs:  # set input placeholder into the lastest layer
            fn_dict = layer_kwargs[key]
            if key in ['act']:
                module_path = fn_dict['module_path']
                func_name = fn_dict['func_name']
                lib = importlib.import_module(module_path)
                fn = getattr(lib, func_name)
                layer_kwargs[key] = fn
                # print(key, layer_kwargs[key])
        # print(name, prev_layer, layer_class, layer_kwargs)

        if layer_class == 'placeholder':  # create placeholder
            if name not in input_list:  # if placeholder is not exist
                dtype = layer_kwargs.pop('dtype')
                shape = layer_kwargs.pop('shape')
                _placeholder = tf.placeholder(eval('tf.' + dtype), shape,
                                              name=name.split(':')[0])  # globals()['tf.'+dtype]
                # _placeholder = tf.placeholder(ast.literal_eval('tf.' + dtype), shape, name=name.split(':')[0])
                # input_dict.update({name: _placeholder})
                input_list.append((name, _placeholder))
        else:  # create network
            if isinstance(prev_layer, list):  # e.g. ConcatLayer, ElementwiseLayer have multiply previous layers
                raise NotImplementedError("TL graph does not support this layer at the moment: %s" % (layer_class))
            else:  # normal layers e.g. Conv2d
                try:  # if previous layer is layer
                    net = layer_dict[prev_layer]
                    layer_kwargs.update({'prev_layer': net})
                except Exception:  # if previous layer is input placeholder
                    for n, t in input_list:
                        if n == prev_layer:
                            _placeholder = t
                    layer_kwargs.update({'inputs': _placeholder})
                layer_kwargs.update({'name': name})
                net = eval('tl.layers.' + layer_class)(**layer_kwargs)
                layer_dict.update({name: net})

    # rename placeholder e.g. x:0 --> x
    for i, (n, t) in enumerate(input_list):
        n_new = n.replace(':', '')
        if n_new[-1] == '0':
            n_new = n_new[:-1]
        input_list[i] = (n_new, t)
        # print(n_new, t)

    # put placeholder into network attributes
    for n, t in input_list:
        # print(name, n, t)
        layer_dict[name].__dict__.update({n: t})
        logging.info("[*] attributes: {} {} {}".format(n, t.get_shape().as_list(), t.dtype.name))
    # for key in input_dict: # set input placeholder into the lastest layer
    #     layer_dict[name].globals()[key] = input_dict[key]
    #     logging.info("  attributes: {:3} {:15} {:15}".format(n, input_dict[key].get_shape().as_list(), input_dict[key].dtype.name))
    logging.info("[*] Load graph finished")
    # return the lastest layer as network
    return layer_dict[name]


def load_graph(name='model.pkl'):
    """Restore TL model archtecture from a a pickle file. No parameters be restored.

    Parameters
    -----------
    name : str
        The name of graph file.

    Returns
    --------
    network : TensorLayer layer
        The input placeholder will become the attributes of the returned TL layer object.

    Examples
    --------
    - see ``tl.files.save_graph``
    """
    logging.info("[*] Loading TL graph from {}".format(name))
    with open(name, 'rb') as file:
        graphs = pickle.load(file)
    return _graph2net(graphs)


def save_graph_and_params(network=None, name='model', sess=None):
    """Save TL model architecture and parameters (i.e. whole model) into graph file and npz file, respectively.

    Parameters
    -----------
    network : TensorLayer layer
        The network to save.
    name : str
        The folder name to save the graph and parameters.
    sess : Session
        TensorFlow Session.

    Examples
    ---------
    Save architecture and parameters

    >>> tl.files.save_graph_and_params(net, 'model', sess)

    Load archtecture and parameters

    >>> net = tl.files.load_graph_and_params('model', sess)
    """
    exists_or_mkdir(name, False)
    save_graph(network, os.path.join(name, 'graph.pkl'))
    save_npz(save_list=network.all_params, name=os.path.join(name, 'params.npz'), sess=sess)


def load_graph_and_params(name='model', sess=None):
    """Load TL model architecture and parameters from graph file and npz file, respectively.

    Parameters
    -----------
    name : str
        The folder name to load the graph and parameters.
    sess : Session
        TensorFlow Session.
    """
    network = load_graph(name=os.path.join(name, 'graph.pkl'))
    load_and_assign_npz(sess=sess, name=os.path.join(name, 'params.npz'), network=network)
    return network
'''


def save_any_to_npy(save_dict=None, name='file.npy'):
    """Save variables to `.npy` file.

    Parameters
    ------------
    save_dict : directory
        The variables to be saved.
    name : str
        File name.

    Examples
    ---------
    >>> tl.files.save_any_to_npy(save_dict={'data': ['a','b']}, name='test.npy')
    >>> data = tl.files.load_npy_to_any(name='test.npy')
    >>> print(data)
    {'data': ['a','b']}

    """
    if save_dict is None:
        save_dict = {}
    np.save(name, save_dict)


def load_npy_to_any(path='', name='file.npy'):
    """Load `.npy` file.

    Parameters
    ------------
    path : str
        Path to the file (optional).
    name : str
        File name.

    Examples
    ---------
    - see tl.files.save_any_to_npy()

    """
    file_path = os.path.join(path, name)
    try:
        return np.load(file_path).item()
    except Exception:
        return np.load(file_path)
    raise Exception("[!] Fail to load %s" % file_path)


def file_exists(filepath):
    """Check whether a file exists by given file path."""
    return os.path.isfile(filepath)


def folder_exists(folderpath):
    """Check whether a folder exists by given folder path."""
    return os.path.isdir(folderpath)


def del_file(filepath):
    """Delete a file by given file path."""
    os.remove(filepath)


def del_folder(folderpath):
    """Delete a folder by given folder path."""
    shutil.rmtree(folderpath)


def read_file(filepath):
    """Read a file and return a string.

    Examples
    ---------
    >>> data = tl.files.read_file('data.txt')

    """
    with open(filepath, 'r') as afile:
        return afile.read()


def load_file_list(path=None, regx='\.jpg', printable=True, keep_prefix=False):
    r"""Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : str or None
        A folder path, if `None`, use the current directory.
    regx : str
        The regx of file name.
    printable : boolean
        Whether to print the files infomation.
    keep_prefix : boolean
        Whether to keep path in the file name.

    Examples
    ----------
    >>> file_list = tl.files.load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')

    """
    if path is None:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if keep_prefix:
        for i, f in enumerate(return_list):
            return_list[i] = os.path.join(path, f)

    if printable:
        logging.info('Match file list = %s' % return_list)
        logging.info('Number of files = %d' % len(return_list))
    return return_list


def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : str
        A folder path.

    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


def exists_or_mkdir(path, verbose=True):
    """Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True.

    Parameters
    ----------
    path : str
        A folder path.
    verbose : boolean
        If True (default), prints results.

    Returns
    --------
    boolean
        True if folder already exist, otherwise, returns False and create the folder.

    Examples
    --------
    >>> tl.files.exists_or_mkdir("checkpoints/train")

    """
    if not os.path.exists(path):
        if verbose:
            logging.info("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            logging.info("[!] %s exists ..." % path)
        return True


def maybe_download_and_extract(filename, working_directory, url_source, extract=False, expected_bytes=None):
    """Checks if file exists in working_directory otherwise tries to dowload the file,
    and optionally also tries to extract the file if format is ".zip" or ".tar"

    Parameters
    -----------
    filename : str
        The name of the (to be) dowloaded file.
    working_directory : str
        A folder path to search for the file in and dowload the file to
    url : str
        The URL to download the file from
    extract : boolean
        If True, tries to uncompress the dowloaded file is ".tar.gz/.tar.bz2" or ".zip" file, default is False.
    expected_bytes : int or None
        If set tries to verify that the downloaded file is of the specified size, otherwise raises an Exception, defaults is None which corresponds to no check being performed.

    Returns
    ----------
    str
        File path of the dowloaded (uncompressed) file.

    Examples
    --------
    >>> down_file = tl.files.maybe_download_and_extract(filename='train-images-idx3-ubyte.gz',
    ...                                            working_directory='data/',
    ...                                            url_source='http://yann.lecun.com/exdb/mnist/')
    >>> tl.files.maybe_download_and_extract(filename='ADEChallengeData2016.zip',
    ...                                             working_directory='data/',
    ...                                             url_source='http://sceneparsing.csail.mit.edu/data/',
    ...                                             extract=True)

    """

    # We first define a download function, supporting both Python 2 and 3.
    def _download(filename, working_directory, url_source):

        progress_bar = progressbar.ProgressBar()

        def _dlProgress(count, blockSize, totalSize, pbar=progress_bar):
            if (totalSize != 0):

                if not pbar.max_value:
                    totalBlocks = math.ceil(float(totalSize) / float(blockSize))
                    pbar.max_value = int(totalBlocks)

                pbar.update(count, force=True)

        filepath = os.path.join(working_directory, filename)

        logging.info('Downloading %s...\n' % filename)

        urlretrieve(url_source + filename, filepath, reporthook=_dlProgress)

    exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):

        _download(filename, working_directory, url_source)
        statinfo = os.stat(filepath)
        logging.info('Succesfully downloaded %s %s bytes.' % (filename, statinfo.st_size))  # , 'bytes.')
        if (not (expected_bytes is None) and (expected_bytes != statinfo.st_size)):
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        if (extract):
            if tarfile.is_tarfile(filepath):
                logging.info('Trying to extract tar file')
                tarfile.open(filepath, 'r').extractall(working_directory)
                logging.info('... Success!')
            elif zipfile.is_zipfile(filepath):
                logging.info('Trying to extract zip file')
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(working_directory)
                logging.info('... Success!')
            else:
                logging.info("Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported")
    return filepath


def natural_keys(text):
    """Sort list of string with number in human order.

    Examples
    ----------
    >>> l = ['im1.jpg', 'im31.jpg', 'im11.jpg', 'im21.jpg', 'im03.jpg', 'im05.jpg']
    >>> l.sort(key=tl.files.natural_keys)
    ['im1.jpg', 'im03.jpg', 'im05', 'im11.jpg', 'im21.jpg', 'im31.jpg']
    >>> l.sort() # that is what we dont want
    ['im03.jpg', 'im05', 'im1.jpg', 'im11.jpg', 'im21.jpg', 'im31.jpg']

    References
    ----------
    - `link <http://nedbatchelder.com/blog/200712/human_sorting.html>`__

    """

    # - alist.sort(key=natural_keys) sorts in human order
    # http://nedbatchelder.com/blog/200712/human_sorting.html
    # (See Toothy's implementation in the comments)
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]


# Visualizing npz files
def npz_to_W_pdf(path=None, regx='w1pre_[0-9]+\.(npz)'):
    r"""Convert the first weight matrix of `.npz` file to `.pdf` by using `tl.visualize.W()`.

    Parameters
    ----------
    path : str
        A folder path to `npz` files.
    regx : str
        Regx for the file name.

    Examples
    ---------
    Convert the first weight matrix of w1_pre...npz file to w1_pre...pdf.

    >>> tl.files.npz_to_W_pdf(path='/Users/.../npz_file/', regx='w1pre_[0-9]+\.(npz)')

    """
    file_list = load_file_list(path=path, regx=regx)
    for f in file_list:
        W = load_npz(path, f)[0]
        logging.info("%s --> %s" % (f, f.split('.')[0] + '.pdf'))
        visualize.draw_weights(W, second=10, saveable=True, name=f.split('.')[0], fig_idx=2012)
