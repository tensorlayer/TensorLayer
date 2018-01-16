#! /usr/bin/python
# -*- coding: utf-8 -*-

import gzip
import os
import pickle
import re
import sys
import tarfile
import zipfile

import numpy as np
import tensorflow as tf
from six.moves import cPickle, urllib, zip
from tensorflow.python.platform import gfile

from . import nlp, utils, visualize


## Load dataset functions
def load_mnist_dataset(shape=(-1, 784), path="data"):
    """Automatically download MNIST dataset
    and return the training, validation and test set with 50000, 10000 and 10000
    digit images respectively.

    Parameters
    ----------
    shape : tuple
        The shape of digit images, defaults is (-1,784)
    path : string
        The path that the data is downloaded to, defaults is ``data/mnist/``.

    Examples
    --------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    path = os.path.join(path, 'mnist')

    # We first define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    def load_mnist_images(path, filename):
        filepath = maybe_download_and_extract(filename, path, 'http://yann.lecun.com/exdb/mnist/')

        print(filepath)
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
        filepath = maybe_download_and_extract(filename, path, 'http://yann.lecun.com/exdb/mnist/')
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # Download and read the training and test set images and labels.
    print("Load or Download MNIST > {}".format(path))
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


def load_cifar10_dataset(shape=(-1, 32, 32, 3), path='data', plotable=False, second=3):
    """The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with
    10000 images. The test batch contains exactly 1000 randomly-selected images from
    each class. The training batches contain the remaining images in random order,
    but some training batches may contain more images from one class than another.
    Between them, the training batches contain exactly 5000 images from each class.

    Parameters
    ----------
    shape : tupe
        The shape of digit images: e.g. (-1, 3, 32, 32) and (-1, 32, 32, 3).
    plotable : True, False
        Whether to plot some image examples.
    second : int
        If ``plotable`` is True, ``second`` is the display time.
    path : string
        The path that the data is downloaded to, defaults is ``data/cifar10/``.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))

    References
    ----------
    - `CIFAR website <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    - `Data download link <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`_
    - `Code references <https://teratail.com/questions/28932>`_
    """
    path = os.path.join(path, 'cifar10')
    print("Load or Download cifar10 > {}".format(path))

    #Helper function to unpickle the data
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
    #Download and uncompress file
    maybe_download_and_extract(filename, path, url, extract=True)

    #Unpickle file and fill in data
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

    if plotable == True:
        print('\nCIFAR-10')
        import matplotlib.pyplot as plt
        fig = plt.figure(1)

        print('Shape of a training image: X_train[0]', X_train[0].shape)

        plt.ion()  # interactive mode
        count = 1
        for row in range(10):
            for col in range(10):
                a = fig.add_subplot(10, 10, count)
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

        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test


def load_ptb_dataset(path='data'):
    """Penn TreeBank (PTB) dataset is used in many LANGUAGE MODELING papers,
    including "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques", "Recurrent Neural Network Regularization".
    It consists of 929k training words, 73k validation words, and 82k test
    words. It has 10k words in its vocabulary.

    Parameters
    ----------
    path : : string
        The path that the data is downloaded to, defaults is ``data/ptb/``.

    Returns
    --------
    train_data, valid_data, test_data, vocabulary size

    Examples
    --------
    >>> train_data, valid_data, test_data, vocab_size = tl.files.load_ptb_dataset()

    References
    ---------------
    - ``tensorflow.models.rnn.ptb import reader``
    - `Manual download <http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz>`_
    """
    path = os.path.join(path, 'ptb')
    print("Load or Download Penn TreeBank (PTB) dataset > {}".format(path))

    #Maybe dowload and uncompress tar, or load exsisting files
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
    vocabulary = len(word_to_id)

    # print(nlp.read_words(train_path))     # ... 'according', 'to', 'mr.', '<unk>', '<eos>']
    # print(train_data)                 # ...  214,         5,    23,    1,       2]
    # print(word_to_id)                 # ... 'beyond': 1295, 'anti-nuclear': 9599, 'trouble': 1520, '<eos>': 2 ... }
    # print(vocabulary)                 # 10000
    # exit()
    return train_data, valid_data, test_data, vocabulary


def load_matt_mahoney_text8_dataset(path='data'):
    """Download a text file from Matt Mahoney's website
    if not present, and make sure it's the right size.
    Extract the first file enclosed in a zip file as a list of words.
    This dataset can be used for Word Embedding.

    Parameters
    ----------
    path : : string
        The path that the data is downloaded to, defaults is ``data/mm_test8/``.

    Returns
    --------
    word_list : a list
        a list of string (word).\n
        e.g. [.... 'their', 'families', 'who', 'were', 'expelled', 'from', 'jerusalem', ...]

    Examples
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> print('Data size', len(words))
    """
    path = os.path.join(path, 'mm_test8')
    print("Load or Download matt_mahoney_text8 Dataset> {}".format(path))

    filename = 'text8.zip'
    url = 'http://mattmahoney.net/dc/'
    maybe_download_and_extract(filename, path, url, expected_bytes=31344016)

    with zipfile.ZipFile(os.path.join(path, filename)) as f:
        word_list = f.read(f.namelist()[0]).split()
        for idx, word in enumerate(word_list):
            word_list[idx] = word_list[idx].decode()
    return word_list


def load_imdb_dataset(path='data', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113, start_char=1, oov_char=2, index_from=3):
    """Load IMDB dataset

    Parameters
    ----------
    path : : string
        The path that the data is downloaded to, defaults is ``data/imdb/``.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(
    ...                                 nb_words=20000, test_split=0.2)
    >>> print('X_train.shape', X_train.shape)
    ... (20000,)  [[1, 62, 74, ... 1033, 507, 27],[1, 60, 33, ... 13, 1053, 7]..]
    >>> print('y_train.shape', y_train.shape)
    ... (20000,)  [1 0 0 ..., 1 0 1]

    References
    -----------
    - `Modified from keras. <https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py>`_
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
        raise Exception('After filtering for sequences shorter than maxlen=' + str(maxlen) + ', no sequence was kept. ' 'Increase maxlen.')
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
    Returns a string.

    Parameters
    ----------
    path : string
        The path that the data is downloaded to, defaults is ``data/nietzsche/``.

    Examples
    --------
    >>> see tutorial_generate_text.py
    >>> words = tl.files.load_nietzsche_dataset()
    >>> words = basic_clean_str(words)
    >>> words = words.split()
    """
    print("Load or Download nietzsche dataset > {}".format(path))
    path = os.path.join(path, 'nietzsche')

    filename = "nietzsche.txt"
    url = 'https://s3.amazonaws.com/text-datasets/'
    filepath = maybe_download_and_extract(filename, path, url)

    with open(filepath, "r") as f:
        words = f.read()
        return words


def load_wmt_en_fr_dataset(path='data'):
    """It will download English-to-French translation data from the WMT'15
    Website (10^9-French-English corpus), and the 2013 news test from
    the same site as development set.
    Returns the directories of training data and test data.

    Parameters
    ----------
    path : string
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
        print("Unpacking %s to %s" % (gz_path, new_path))
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
            print("Extracting tgz file %s" % dev_file)
            with tarfile.open(dev_file, "r:gz") as dev_tar:
                fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
                en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
                fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
                en_dev_file.name = dev_name + ".en"
                dev_tar.extract(fr_dev_file, path)
                dev_tar.extract(en_dev_file, path)
        return dev_path

    print("Load or Download WMT English-to-French translation > {}".format(path))

    train_path = get_wmt_enfr_train_set(path)
    dev_path = get_wmt_enfr_dev_set(path)

    return train_path, dev_path


def load_flickr25k_dataset(tag='sky', path="data", n_threads=50, printable=False):
    """Returns a list of images by a given tag from Flick25k dataset,
    it will download Flickr25k from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`_
    at the first time you use it.

    Parameters
    ------------
    tag : string or None
        If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`_.
        If you want to get all images, set to ``None``.
    path : string
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int, number of thread to read image.
    printable : bool, print infomation when reading images, default is ``False``.

    Examples
    -----------
    - Get images with tag of sky
    >>> images = tl.files.load_flickr25k_dataset(tag='sky')

    - Get all images
    >>> images = tl.files.load_flickr25k_dataset(tag=None, n_threads=100, printable=True)
    """
    path = os.path.join(path, 'flickr25k')

    filename = 'mirflickr25k.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr25k/'
    ## download dataset
    if folder_exists(path + "/mirflickr") is False:
        print("[*] Flickr25k is nonexistent in {}".format(path))
        maybe_download_and_extract(filename, path, url, extract=True)
        del_file(path + '/' + filename)
    ## return images by the given tag.
    # 1. image path list
    folder_imgs = path + "/mirflickr"
    path_imgs = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    path_imgs.sort(key=natural_keys)
    # print(path_imgs[0:10])
    # 2. tag path list
    folder_tags = path + "/mirflickr/meta/tags"
    path_tags = load_file_list(path=folder_tags, regx='\\.txt', printable=False)
    path_tags.sort(key=natural_keys)
    # print(path_tags[0:10])
    # 3. select images
    if tag is None:
        print("[Flickr25k] reading all images")
    else:
        print("[Flickr25k] reading images with tag: {}".format(tag))
    images_list = []
    for idx in range(0, len(path_tags)):
        tags = read_file(folder_tags + '/' + path_tags[idx]).split('\n')
        # print(idx+1, tags)
        if tag is None or tag in tags:
            images_list.append(path_imgs[idx])

    images = visualize.read_images(images_list, folder_imgs, n_threads=n_threads, printable=printable)
    return images


def load_flickr1M_dataset(tag='sky', size=10, path="data", n_threads=50, printable=False):
    """Returns a list of images by a given tag from Flickr1M dataset,
    it will download Flickr1M from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`_
    at the first time you use it.

    Parameters
    ------------
    tag : string or None
        If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`_.
        If you want to get all images, set to ``None``.
    size : int 1 to 10.
        1 means 100k images ... 5 means 500k images, 10 means all 1 million images. Default is 10.
    path : string
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int, number of thread to read image.
    printable : bool, print infomation when reading images, default is ``False``.

    Examples
    ----------
    - Use 200k images
    >>> images = tl.files.load_flickr1M_dataset(tag='zebra', size=2)

    - Use 1 Million images
    >>> images = tl.files.load_flickr1M_dataset(tag='zebra')
    """
    path = os.path.join(path, 'flickr1M')
    print("[Flickr1M] using {}% of images = {}".format(size * 10, size * 100000))
    images_zip = [
        'images0.zip', 'images1.zip', 'images2.zip', 'images3.zip', 'images4.zip', 'images5.zip', 'images6.zip', 'images7.zip', 'images8.zip', 'images9.zip'
    ]
    tag_zip = 'tags.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr1m/'
    ## download dataset
    for image_zip in images_zip[0:size]:
        image_folder = image_zip.split(".")[0]
        # print(path+"/"+image_folder)
        if folder_exists(path + "/" + image_folder) is False:
            # print(image_zip)
            print("[Flickr1M] {} is missing in {}".format(image_folder, path))
            maybe_download_and_extract(image_zip, path, url, extract=True)
            del_file(path + '/' + image_zip)
            os.system("mv {} {}".format(path + '/images', path + '/' + image_folder))
        else:
            print("[Flickr1M] {} exists in {}".format(image_folder, path))
    ## download tag
    if folder_exists(path + "/tags") is False:
        print("[Flickr1M] tag files is nonexistent in {}".format(path))
        maybe_download_and_extract(tag_zip, path, url, extract=True)
        del_file(path + '/' + tag_zip)
    else:
        print("[Flickr1M] tags exists in {}".format(path))

    ## 1. image path list
    images_list = []
    images_folder_list = []
    for i in range(0, size):
        images_folder_list += load_folder_list(path=path + '/images%d' % i)
    images_folder_list.sort(key=lambda s: int(s.split('/')[-1]))  # folder/images/ddd
    # print(images_folder_list)
    # exit()
    for folder in images_folder_list[0:size * 10]:
        tmp = load_file_list(path=folder, regx='\\.jpg', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.jpg
        # print(tmp[0::570])
        images_list.extend([folder + '/' + x for x in tmp])
    # print('IM', len(images_list), images_list[0::6000])
    ## 2. tag path list
    tag_list = []
    tag_folder_list = load_folder_list(path + "/tags")
    tag_folder_list.sort(key=lambda s: int(s.split('/')[-1]))  # folder/images/ddd

    for folder in tag_folder_list[0:size * 10]:
        # print(folder)
        tmp = load_file_list(path=folder, regx='\\.txt', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.txt
        tmp = [folder + '/' + s for s in tmp]
        tag_list += tmp
    # print('T', len(tag_list), tag_list[0::6000])
    # exit()
    ## 3. select images
    print("[Flickr1M] searching tag: {}".format(tag))
    select_images_list = []
    for idx in range(0, len(tag_list)):
        tags = read_file(tag_list[idx]).split('\n')
        if tag in tags:
            select_images_list.append(images_list[idx])
            # print(idx, tags, tag_list[idx], images_list[idx])
    print("[Flickr1M] reading images with tag: {}".format(tag))
    images = visualize.read_images(select_images_list, '', n_threads=n_threads, printable=printable)
    return images


def load_cyclegan_dataset(filename='summer2winter_yosemite', path='data'):
    """Load image data from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`_.

    Parameters
    ------------
    filename : string
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`_.
    path : string
        The path that the data is downloaded to, defaults is `data/cyclegan`

    Examples
    ---------
    >>> im_train_A, im_train_B, im_test_A, im_test_B = load_cyclegan_dataset(filename='summer2winter_yosemite')
    """
    path = os.path.join(path, 'cyclegan')
    url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

    if folder_exists(os.path.join(path, filename)) is False:
        print("[*] {} is nonexistent in {}".format(filename, path))
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
        for i in range(len(images)):
            if len(images[i].shape) == 2:
                images[i] = images[i][:, :, np.newaxis]
                images[i] = np.tile(images[i], (1, 1, 3))
        return images

    im_train_A = if_2d_to_3d(im_train_A)
    im_train_B = if_2d_to_3d(im_train_B)
    im_test_A = if_2d_to_3d(im_test_A)
    im_test_B = if_2d_to_3d(im_test_B)

    return im_train_A, im_train_B, im_test_A, im_test_B


def download_file_from_google_drive(id, destination):
    """ Download file from Google Drive, see ``load_celebA_dataset`` for example.

    Parameters
    --------------
    id : driver ID
    destination : string, save path.
    """
    from tqdm import tqdm
    import requests

    def save_response_content(response, destination, chunk_size=32 * 1024):
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_scale=True, desc=destination):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def load_celebA_dataset(dirpath='data'):
    """ Automatically download celebA dataset, and return a list of image path. """
    import zipfile, os
    data_dir = 'celebA'
    filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(dirpath, filename)
    image_path = os.path.join(dirpath, data_dir)
    if os.path.exists(image_path):
        print('[*] {} already exists'.format(save_path))
    else:
        exists_or_mkdir(dirpath)
        download_file_from_google_drive(drive_id, save_path)
        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(save_path)
        os.rename(os.path.join(dirpath, zip_dir), image_path)

    data_files = load_file_list(path=image_path, regx='\\.jpg', printable=False)
    for i in range(len(data_files)):
        data_files[i] = os.path.join(image_path, data_files[i])
    return data_files


def load_voc_dataset(path='data', dataset='2012', contain_classes_in_person=False):
    """ Pascal VOC 2007/2012 Dataset has 20 objects : aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor and additional 3 classes : head, hand, foot for person.

    Parameters
    -----------
    path : string
        The path that the data is downloaded to, defaults is ``data/VOC``.
    dataset : string, 2012, 2007, 2007test or 2012test.
        The VOC dataset version, we usually train model on 2007+2012 and test it on 2007test.
    contain_classes_in_person : If True, dataset will contains labels of head, hand and foot.

    Returns
    ---------
    imgs_file_list : list of string.
        Full paths of all images.
    imgs_semseg_file_list : list of string.
        Full paths of all maps for semantic segmentation. Note that not all images have this map!
    imgs_insseg_file_list : list of string.
        Full paths of all maps for instance segmentation. Note that not all images have this map!
    imgs_ann_file_list : list of string.
        Full paths of all annotations for bounding box and object class, all images have this annotations.
    classes : list of string.
        Classes in order.
    classes_in_person : list of string.
        Classes in person.
    classes_dict : dictionary.
        Class label to integer.
    n_objs_list : list of integer
        Number of objects in all images in ``imgs_file_list` in order.
    objs_info_list : list of string.
        Darknet format for the annotation of all images in ``imgs_file_list`` in order. ``[class_id x_centre y_centre width height]`` in ratio format.
    objs_info_dicts : dictionary.
        ``{imgs_file_list : dictionary for annotation}``, the annotation of all images in ``imgs_file_list``,
        format from `TensorFlow/Models/object-detection <https://github.com/tensorflow/models/blob/master/object_detection/create_pascal_tf_record.py>`_.

    Examples
    ----------
    >>> imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list,
    >>>     classes, classes_in_person, classes_dict,
    >>>     n_objs_list, objs_info_list, objs_info_dicts = tl.files.load_voc_dataset(dataset="2012", contain_classes_in_person=False)
    >>> idx = 26
    >>> print(classes)
    ... ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    >>> print(classes_dict)
    ... {'sheep': 16, 'horse': 12, 'bicycle': 1, 'bottle': 4, 'cow': 9, 'sofa': 17, 'car': 6, 'dog': 11, 'cat': 7, 'person': 14, 'train': 18, 'diningtable': 10, 'aeroplane': 0, 'bus': 5, 'pottedplant': 15, 'tvmonitor': 19, 'chair': 8, 'bird': 2, 'boat': 3, 'motorbike': 13}
    >>> print(imgs_file_list[idx])
    ... data/VOC/VOC2012/JPEGImages/2007_000423.jpg
    >>> print(n_objs_list[idx])
    ... 2
    >>> print(imgs_ann_file_list[idx])
    ... data/VOC/VOC2012/Annotations/2007_000423.xml
    >>> print(objs_info_list[idx])
    ... 14 0.173 0.461333333333 0.142 0.496
    ... 14 0.828 0.542666666667 0.188 0.594666666667
    >>> ann = tl.prepro.parse_darknet_ann_str_to_list(objs_info_list[idx])
    >>> print(ann)
    ... [[14, 0.173, 0.461333333333, 0.142, 0.496], [14, 0.828, 0.542666666667, 0.188, 0.594666666667]]
    >>> c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    >>> print(c, b)
    ... [14, 14] [[0.173, 0.461333333333, 0.142, 0.496], [0.828, 0.542666666667, 0.188, 0.594666666667]]

    References
    -------------
    - `Pascal VOC2012 Website <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_.
    - `Pascal VOC2007 Website <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/>`_.
    - `TensorFlow/Models/object-detection <https://github.com/zsdonghao/object-detection/blob/master/g3doc/preparing_inputs.md>`_.
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

    from lxml import etree  # pip install lxml
    import xml.etree.ElementTree as ET

    ##
    if dataset == "2012":
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
        tar_filename = "VOCtrainval_11-May-2012.tar"
        extracted_filename = "VOC2012"  #"VOCdevkit/VOC2012"
        print("    [============= VOC 2012 =============]")
    elif dataset == "2012test":
        extracted_filename = "VOC2012test"  #"VOCdevkit/VOC2012"
        print("    [============= VOC 2012 Test Set =============]")
        print("    \nAuthor: 2012test only have person annotation, so 2007test is highly recommended for testing !\n")
        import time
        time.sleep(3)
        if os.path.isdir(os.path.join(path, extracted_filename)) is False:
            print("For VOC 2012 Test data - online registration required")
            print(
                " Please download VOC2012test.tar from:  \n register: http://host.robots.ox.ac.uk:8080 \n voc2012 : http://host.robots.ox.ac.uk:8080/eval/challenges/voc2012/ \ndownload: http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar"
            )
            print(" unzip VOC2012test.tar,rename the folder to VOC2012test and put it into %s" % path)
            exit()
        # # http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2012test.tar
        # url = "http://host.robots.ox.ac.uk:8080/eval/downloads/"
        # tar_filename = "VOC2012test.tar"
    elif dataset == "2007":
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        tar_filename = "VOCtrainval_06-Nov-2007.tar"
        extracted_filename = "VOC2007"
        print("    [============= VOC 2007 =============]")
    elif dataset == "2007test":
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html#testdata
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/"
        tar_filename = "VOCtest_06-Nov-2007.tar"
        extracted_filename = "VOC2007test"
        print("    [============= VOC 2007 Test Set =============]")
    else:
        raise Exception("Please set the dataset aug to 2012, 2012test or 2007.")

    ##======== download dataset
    if dataset != "2012test":
        from sys import platform as _platform
        if folder_exists(os.path.join(path, extracted_filename)) is False:
            print("[VOC] {} is nonexistent in {}".format(extracted_filename, path))
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
    ##======== object classes(labels)  NOTE: YOU CAN CUSTOMIZE THIS LIST
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    if contain_classes_in_person:
        classes_in_person = ["head", "hand", "foot"]
    else:
        classes_in_person = []

    classes += classes_in_person  # use extra 3 classes for person

    classes_dict = utils.list_string_to_dict(classes)
    print("[VOC] object classes {}".format(classes_dict))

    ##======== 1. image path list
    # folder_imgs = path+"/"+extracted_filename+"/JPEGImages/"
    folder_imgs = os.path.join(path, extracted_filename, "JPEGImages")
    imgs_file_list = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    print("[VOC] {} images found".format(len(imgs_file_list)))
    imgs_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2]))  # 2007_000027.jpg --> 2007000027
    imgs_file_list = [os.path.join(folder_imgs, s) for s in imgs_file_list]
    # print('IM',imgs_file_list[0::3333], imgs_file_list[-1])
    if dataset != "2012test":
        ##======== 2. semantic segmentation maps path list
        # folder_semseg = path+"/"+extracted_filename+"/SegmentationClass/"
        folder_semseg = os.path.join(path, extracted_filename, "SegmentationClass")
        imgs_semseg_file_list = load_file_list(path=folder_semseg, regx='\\.png', printable=False)
        print("[VOC] {} maps for semantic segmentation found".format(len(imgs_semseg_file_list)))
        imgs_semseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2]))  # 2007_000032.png --> 2007000032
        imgs_semseg_file_list = [os.path.join(folder_semseg, s) for s in imgs_semseg_file_list]
        # print('Semantic Seg IM',imgs_semseg_file_list[0::333], imgs_semseg_file_list[-1])
        ##======== 3. instance segmentation maps path list
        # folder_insseg = path+"/"+extracted_filename+"/SegmentationObject/"
        folder_insseg = os.path.join(path, extracted_filename, "SegmentationObject")
        imgs_insseg_file_list = load_file_list(path=folder_insseg, regx='\\.png', printable=False)
        print("[VOC] {} maps for instance segmentation found".format(len(imgs_semseg_file_list)))
        imgs_insseg_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2]))  # 2007_000032.png --> 2007000032
        imgs_insseg_file_list = [os.path.join(folder_insseg, s) for s in imgs_insseg_file_list]
        # print('Instance Seg IM',imgs_insseg_file_list[0::333], imgs_insseg_file_list[-1])
    else:
        imgs_semseg_file_list = []
        imgs_insseg_file_list = []
    ##======== 4. annotations for bounding box and object class
    # folder_ann = path+"/"+extracted_filename+"/Annotations/"
    folder_ann = os.path.join(path, extracted_filename, "Annotations")
    imgs_ann_file_list = load_file_list(path=folder_ann, regx='\\.xml', printable=False)
    print("[VOC] {} XML annotation files for bounding box and object class found".format(len(imgs_ann_file_list)))
    imgs_ann_file_list.sort(key=lambda s: int(s.replace('.', ' ').replace('_', '').split(' ')[-2]))  # 2007_000027.xml --> 2007000027
    imgs_ann_file_list = [os.path.join(folder_ann, s) for s in imgs_ann_file_list]
    # print('ANN',imgs_ann_file_list[0::3333], imgs_ann_file_list[-1])

    if dataset == "2012test":  # remove unused images in JPEG folder
        imgs_file_list_new = []
        for ann in imgs_ann_file_list:
            ann = os.path.split(ann)[-1].split('.')[0]
            for im in imgs_file_list:
                if ann in im:
                    imgs_file_list_new.append(im)
                    break
        imgs_file_list = imgs_file_list_new
        print("[VOC] keep %d images" % len(imgs_file_list_new))

    ##======== parse XML annotations
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
        return (x, y, w, h)

    def convert_annotation(file_name):
        """ Given VOC2012 XML Annotations, returns number of objects and info. """
        in_file = open(file_name)
        out_file = ""
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        n_objs = 0

        # print(file_name, w, h, size)
        # exit()
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
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
            n_objs += 1
            if cls in "person":
                for part in obj.iter('part'):
                    cls = part.find('name').text
                    if cls not in classes_in_person:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = part.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((w, h), b)
                    # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    out_file += str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                    n_objs += 1
        in_file.close()
        return n_objs, out_file

    print("[VOC] Parsing xml annotations files")
    n_objs_list = []
    objs_info_list = []  # Darknet Format list of string
    objs_info_dicts = {}
    for idx, ann_file in enumerate(imgs_ann_file_list):
        # print(ann_file)
        n_objs, objs_info = convert_annotation(ann_file)
        n_objs_list.append(n_objs)
        objs_info_list.append(objs_info)
        with tf.gfile.GFile(ann_file, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = _recursive_parse_xml_to_dict(xml)['annotation']
        objs_info_dicts.update({imgs_file_list[idx]: data})

    return imgs_file_list, imgs_semseg_file_list, imgs_insseg_file_list, imgs_ann_file_list, \
        classes, classes_in_person, classes_dict,\
        n_objs_list, objs_info_list, objs_info_dicts


## Load and save network list npz
def save_npz(save_list=[], name='model.npz', sess=None):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.

    Parameters
    ----------
    save_list : a list
        Parameters want to be saved.
    name : a string or None
        The name of the .npz file.
    sess : None or Session

    Examples
    --------
    - Save model to npz
    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
    - Load model from npz (Method 1)
    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    - Load model from npz (Method 2)
    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    Notes
    -----
    If you got session issues, you can change the value.eval() to value.eval(session=sess)

    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## save params into a list
    save_list_var = []
    if sess:
        save_list_var = sess.run(save_list)
    else:
        try:
            for k, value in enumerate(save_list):
                save_list_var.append(value.eval())
        except:
            print(" Fail to save model, Hint: pass the session into this function, save_npz(network.all_params, name='model.npz', sess=sess)")
    np.savez(name, params=save_list_var)
    save_list_var = None
    del save_list_var
    print("[*] %s saved" % name)

    ## save params into a dictionary
    # rename_dict = {}
    # for k, value in enumerate(save_dict):
    #     rename_dict.update({'param'+str(k) : value.eval()})
    # np.savez(name, **rename_dict)
    # print('Model is saved to: %s' % name)


def load_npz(path='', name='model.npz'):
    """Load the parameters of a Model saved by tl.files.save_npz().

    Parameters
    ----------
    path : a string
        Folder path to .npz file.
    name : a string or None
        The name of the .npz file.

    Returns
    --------
    params : list
        A list of parameters in order.

    Examples
    --------
    - See ``save_npz``

    References
    ----------
    - `Saving dictionary using numpy <http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez>`_
    """
    ## if save_npz save params into a dictionary
    # d = np.load( path+name )
    # params = []
    # print('Load Model')
    # for key, val in sorted( d.items() ):
    #     params.append(val)
    #     print('Loading %s, %s' % (key, str(val.shape)))
    # return params
    ## if save_npz save params into a list
    d = np.load(path + name)
    # for val in sorted( d.items() ):
    #     params = val
    #     return params
    return d['params']
    # print(d.items()[0][1]['params'])
    # exit()
    # return d.items()[0][1]['params']


def assign_params(sess, params, network):
    """Assign the given parameters to the TensorLayer network.

    Parameters
    ----------
    sess : TensorFlow Session. Automatically run when sess is not None.
    params : a list
        A list of parameters in order.
    network : a :class:`Layer` class
        The network to be assigned

    Returns
    --------
    ops : list
        A list of tf ops in order that assign params. Support sess.run(ops) manually.

    Examples
    --------
    - Save model to npz
    >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
    - Load model from npz (Method 1)
    >>> load_params = tl.files.load_npz(name='model.npz')
    >>> tl.files.assign_params(sess, load_params, network)
    - Load model from npz (Method 2)
    >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    References
    ----------
    - `Assign value to a TensorFlow variable <http://stackoverflow.com/questions/34220532/how-to-assign-value-to-a-tensorflow-variable>`_
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
    sess : TensorFlow Session
    name : string
        Model path.
    network : a :class:`Layer` class
        The network to be assigned

    Returns
    --------
    Returns False if faild to model is not exist.

    Examples
    ---------
    >>> tl.files.save_npz(net.all_params, name='net.npz', sess=sess)
    >>> tl.files.load_and_assign_npz(sess=sess, name='net.npz', network=net)
    """
    assert network is not None
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Load {} failed!".format(name))
        return False
    else:
        params = load_npz(name=name)
        assign_params(sess, params, network)
        print("[*] Load {} SUCCESS!".format(name))
        return network


## Load and save network dict npz
def save_npz_dict(save_list=[], name='model.npz', sess=None):
    """Input parameters and the file name, save parameters as a dictionary into .npz file.
    Use ``tl.files.load_and_assign_npz_dict()`` to restore.

    Parameters
    ----------
    save_list : a list to tensor for parameters
        Parameters want to be saved.
    name : a string
        The name of the .npz file.
    sess : Session
    """
    assert sess is not None
    save_list_names = [tensor.name for tensor in save_list]
    save_list_var = sess.run(save_list)
    save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
    np.savez(name, **save_var_dict)
    save_list_var = None
    save_var_dict = None
    del save_list_var
    del save_var_dict
    print("[*] Model saved in npz_dict %s" % name)


def load_and_assign_npz_dict(name='model.npz', sess=None):
    """Restore the parameters saved by ``tl.files.save_npz_dict()``.

    Parameters
    ----------
    name : a string
        The name of the .npz file.
    sess : Session
    """
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Load {} failed!".format(name))
        return False

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
                print("[*] params restored: %s" % key)
        except KeyError:
            print("[!] Warning: Tensor named %s not found in network." % key)

    sess.run(ops)
    print("[*] Model restored from npz_dict %s" % name)


# def save_npz_dict(save_list=[], name='model.npz', sess=None):
#     """Input parameters and the file name, save parameters as a dictionary into .npz file. Use tl.utils.load_npz_dict() to restore.
#
#     Parameters
#     ----------
#     save_list : a list
#         Parameters want to be saved.
#     name : a string or None
#         The name of the .npz file.
#     sess : None or Session
#
#     Notes
#     -----
#     This function tries to avoid a potential broadcasting error raised by numpy.
#
#     """
#     ## save params into a list
#     save_list_var = []
#     if sess:
#         save_list_var = sess.run(save_list)
#     else:
#         try:
#             for k, value in enumerate(save_list):
#                 save_list_var.append(value.eval())
#         except:
#             print(" Fail to save model, Hint: pass the session into this function, save_npz_dict(network.all_params, name='model.npz', sess=sess)")
#     save_var_dict = {str(idx):val for idx, val in enumerate(save_list_var)}
#     np.savez(name, **save_var_dict)
#     save_list_var = None
#     save_var_dict = None
#     del save_list_var
#     del save_var_dict
#     print("[*] %s saved" % name)
#
# def load_npz_dict(path='', name='model.npz'):
#     """Load the parameters of a Model saved by tl.files.save_npz_dict().
#
#     Parameters
#     ----------
#     path : a string
#         Folder path to .npz file.
#     name : a string or None
#         The name of the .npz file.
#
#     Returns
#     --------
#     params : list
#         A list of parameters in order.
#     """
#     d = np.load( path+name )
#     saved_list_var = [val[1] for val in sorted(d.items(), key=lambda tup: int(tup[0]))]
#     return saved_list_var


## Load and save network ckpt
def save_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], global_step=None, printable=False):
    """Save parameters into ckpt file.

    Parameters
    ------------
    sess : Session.
    mode_name : string, name of the model, default is ``model.ckpt``.
    save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
    var_list : list of variables, if not given, save all global variables.
    global_step : int or None, step number.
    printable : bool, if True, print all params info.

    Examples
    ---------
    - see ``tl.files.load_ckpt()``.
    """
    assert sess is not None
    ckpt_file = os.path.join(save_dir, mode_name)
    if var_list == []:
        var_list = tf.global_variables()

    print("[*] save %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    saver = tf.train.Saver(var_list)
    saver.save(sess, ckpt_file, global_step=global_step)


def load_ckpt(sess=None, mode_name='model.ckpt', save_dir='checkpoint', var_list=[], is_latest=True, printable=False):
    """Load parameters from ckpt file.

    Parameters
    ------------
    sess : Session.
    mode_name : string, name of the model, default is ``model.ckpt``.
        Note that if ``is_latest`` is True, this function will get the ``mode_name`` automatically.
    save_dir : string, path / file directory to the ckpt, default is ``checkpoint``.
    var_list : list of variables, if not given, save all global variables.
    is_latest : bool, if True, load the latest ckpt, if False, load the ckpt with the name of ```mode_name``.
    printable : bool, if True, print all params info.

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
    assert sess is not None

    if is_latest:
        ckpt_file = tf.train.latest_checkpoint(save_dir)
    else:
        ckpt_file = os.path.join(save_dir, mode_name)

    if var_list == []:
        var_list = tf.global_variables()

    print("[*] load %s n_params: %d" % (ckpt_file, len(var_list)))

    if printable:
        for idx, v in enumerate(var_list):
            print("  param {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    try:
        saver = tf.train.Saver(var_list)
        saver.restore(sess, ckpt_file)
    except Exception as e:
        print(e)
        print("[*] load ckpt fail ...")


## Load and save variables
def save_any_to_npy(save_dict={}, name='file.npy'):
    """Save variables to .npy file.

    Examples
    ---------
    >>> tl.files.save_any_to_npy(save_dict={'data': ['a','b']}, name='test.npy')
    >>> data = tl.files.load_npy_to_any(name='test.npy')
    >>> print(data)
    ... {'data': ['a','b']}
    """
    np.save(name, save_dict)


def load_npy_to_any(path='', name='file.npy'):
    """Load .npy file.

    Examples
    ---------
    - see save_any_to_npy()
    """
    file_path = os.path.join(path, name)
    try:
        npy = np.load(file_path).item()
    except:
        npy = np.load(file_path)
    finally:
        try:
            return npy
        except:
            print("[!] Fail to load %s" % file_path)
            exit()


## Folder functions
def file_exists(filepath):
    """ Check whether a file exists by given file path. """
    return os.path.isfile(filepath)


def folder_exists(folderpath):
    """ Check whether a folder exists by given folder path. """
    return os.path.isdir(folderpath)


def del_file(filepath):
    """ Delete a file by given file path. """
    os.remove(filepath)


def del_folder(folderpath):
    """ Delete a folder by given folder path. """
    os.rmdir(folderpath)


def read_file(filepath):
    """ Read a file and return a string.

    Examples
    ---------
    >>> data = tl.files.read_file('data.txt')
    """
    with open(filepath, 'r') as afile:
        return afile.read()


def load_file_list(path=None, regx='\.npz', printable=True):
    """Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : a string or None
        A folder path.
    regx : a string
        The regx of file name.
    printable : boolean, whether to print the files infomation.

    Examples
    ----------
    >>> file_list = tl.files.load_file_list(path=None, regx='w1pre_[0-9]+\.(npz)')
    """
    if path == False:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list


def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : a string or None
        A folder path.
    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


def exists_or_mkdir(path, verbose=True):
    """Check a folder by given name, if not exist, create the folder and return False,
    if directory exists, return True.

    Parameters
    ----------
    path : a string
        A folder path.
    verbose : boolean
        If True, prints results, deaults is True

    Returns
    --------
    True if folder exist, otherwise, returns False and create the folder

    Examples
    --------
    >>> tl.files.exists_or_mkdir("checkpoints/train")
    """
    if not os.path.exists(path):
        if verbose:
            print("[*] creates %s ..." % path)
        os.makedirs(path)
        return False
    else:
        if verbose:
            print("[!] %s exists ..." % path)
        return True


def maybe_download_and_extract(filename, working_directory, url_source, extract=False, expected_bytes=None):
    """Checks if file exists in working_directory otherwise tries to dowload the file,
    and optionally also tries to extract the file if format is ".zip" or ".tar"

    Parameters
    -----------
    filename : string
        The name of the (to be) dowloaded file.
    working_directory : string
        A folder path to search for the file in and dowload the file to
    url : string
        The URL to download the file from
    extract : bool, defaults is False
        If True, tries to uncompress the dowloaded file is ".tar.gz/.tar.bz2" or ".zip" file
    expected_bytes : int/None
        If set tries to verify that the downloaded file is of the specified size, otherwise raises an Exception,
        defaults is None which corresponds to no check being performed

    Returns
    ----------
    filepath to dowloaded (uncompressed) file

    Examples
    --------
    >>> down_file = tl.files.maybe_download_and_extract(filename = 'train-images-idx3-ubyte.gz',
                                                        working_directory = 'data/',
                                                        url_source = 'http://yann.lecun.com/exdb/mnist/')
    >>> tl.files.maybe_download_and_extract(filename = 'ADEChallengeData2016.zip',
                                            working_directory = 'data/',
                                            url_source = 'http://sceneparsing.csail.mit.edu/data/',
                                            extract=True)
    """

    # We first define a download function, supporting both Python 2 and 3.
    def _download(filename, working_directory, url_source):
        def _dlProgress(count, blockSize, totalSize):
            if (totalSize != 0):
                percent = float(count * blockSize) / float(totalSize) * 100.0
                sys.stdout.write("\r" "Downloading " + filename + "...%d%%" % percent)
                sys.stdout.flush()

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve
        filepath = os.path.join(working_directory, filename)
        urlretrieve(url_source + filename, filepath, reporthook=_dlProgress)

    exists_or_mkdir(working_directory, verbose=False)
    filepath = os.path.join(working_directory, filename)

    if not os.path.exists(filepath):
        _download(filename, working_directory, url_source)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded %s %s bytes.' % (filename, statinfo.st_size))  #, 'bytes.')
        if (not (expected_bytes is None) and (expected_bytes != statinfo.st_size)):
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        if (extract):
            if tarfile.is_tarfile(filepath):
                print('Trying to extract tar file')
                tarfile.open(filepath, 'r').extractall(working_directory)
                print('... Success!')
            elif zipfile.is_zipfile(filepath):
                print('Trying to extract zip file')
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(working_directory)
                print('... Success!')
            else:
                print("Unknown compression_format only .tar.gz/.tar.bz2/.tar and .zip supported")
    return filepath


## Sort
def natural_keys(text):
    """Sort list of string with number in human order.

    Examples
    ----------
    >>> l = ['im1.jpg', 'im31.jpg', 'im11.jpg', 'im21.jpg', 'im03.jpg', 'im05.jpg']
    >>> l.sort(key=tl.files.natural_keys)
    ... ['im1.jpg', 'im03.jpg', 'im05', 'im11.jpg', 'im21.jpg', 'im31.jpg']
    >>> l.sort() # that is what we dont want
    ... ['im03.jpg', 'im05', 'im1.jpg', 'im11.jpg', 'im21.jpg', 'im31.jpg']

    References
    ----------
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split('(\d+)', text)]


# Visualizing npz files
def npz_to_W_pdf(path=None, regx='w1pre_[0-9]+\.(npz)'):
    """Convert the first weight matrix of .npz file to .pdf by using tl.visualize.W().

    Parameters
    ----------
    path : a string or None
        A folder path to npz files.
    regx : a string
        Regx for the file name.

    Examples
    --------
    >>> Convert the first weight matrix of w1_pre...npz file to w1_pre...pdf.
    >>> tl.files.npz_to_W_pdf(path='/Users/.../npz_file/', regx='w1pre_[0-9]+\.(npz)')
    """
    file_list = load_file_list(path=path, regx=regx)
    for f in file_list:
        W = load_npz(path, f)[0]
        print("%s --> %s" % (f, f.split('.')[0] + '.pdf'))
        visualize.W(W, second=10, saveable=True, name=f.split('.')[0], fig_idx=2012)
