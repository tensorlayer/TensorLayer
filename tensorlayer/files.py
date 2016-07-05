#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import os
import numpy as np
import sys
import re
import tensorlayer.visualize as visualize
import collections
from six.moves import xrange

## Load Data
def load_mnist_dataset(shape=(-1,784)):
    """Automatically download MNIST dataset
    and return the training, validation and test set with 50000, 10000 and 10000
    digit images respectively.

    Parameters
    ----------
    shape : tuple
        The shape of digit images

    Examples
    --------
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))
    """
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(shape)
        # data = data.reshape(-1, 1, 28, 28)    # for lasagne
        # data = data.reshape(-1, 28, 28, 1)      # for tensorflow
        # data = data.reshape(-1, 784)      # for tensorflow
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    ## you may want to change the path
    data_dir = ''   #os.getcwd() + '/lasagne_tutorial/'
    # print('data_dir > %s' % data_dir)

    X_train = load_mnist_images(data_dir+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_dir+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    ## you may want to plot one example
    # print('X_train[0][0] >', X_train[0][0].shape, type(X_train[0][0]))  # for lasagne
    # print('X_train[0] >', X_train[0].shape, type(X_train[0]))       # for tensorflow
    # # exit()
    #         #  [[..],[..]]      (28, 28)      numpy.ndarray
    #         # plt.imshow 只支持 (28, 28)格式，不支持 (1, 28, 28),所以用 [0][0]
    # fig = plt.figure()
    # #plotwindow = fig.add_subplot(111)
    # # plt.imshow(X_train[0][0], cmap='gray')    # for lasagne (-1, 1, 28, 28)
    # plt.imshow(X_train[0].reshape(28,28), cmap='gray')     # for tensorflow (-1, 28, 28, 1)
    # plt.title('A training image')
    # plt.show()

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False, second=3):
    """The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with
    6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with
    10000 images. The test batch contains exactly 1000 randomly-selected images from
    each class. The training batches contain the remaining images in random order,
    but some training batches may contain more images from one class than another.
    Between them, the training batches contain exactly 5000 images from each class.

    # code references : https://teratail.com/questions/28932
    # data download link : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
                           https://www.cs.toronto.edu/~kriz/cifar.html

    Parameters
    ----------
    shape : tupe
        The shape of digit images: e.g. (-1, 3, 32, 32) , (-1, 32, 32, 3) , (-1, 32*32*3)
    plotable : True, False
        Whether to plot some image examples.
    second : int
        If 'plotable' is True, 'second' is the display time.

    Examples
    --------
    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=True)

    Note
    ------
    CIFAR-10 images can only be display without color change under uint8.
    >>> X_train = np.asarray(X_train, dtype=np.uint8)
    >>> plt.ion()
    >>> fig = plt.figure(1232)
    >>> count = 1
    >>> for row in range(10):
    >>>     for col in range(10):
    >>>         a = fig.add_subplot(10, 10, count)
    >>>         plt.imshow(X_train[count-1], interpolation='nearest')
    >>>         plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
    >>>         plt.gca().yaxis.set_major_locator(plt.NullLocator())
    >>>         count = count + 1
    >>> plt.draw()
    >>> plt.pause(3)
    """
    import sys
    import pickle
    import numpy as np


    # We first define a download function, supporting both Python 2 and 3.
    filename = 'cifar-10-python.tar.gz'
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='https://www.cs.toronto.edu/~kriz/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # After downloading the cifar-10-python.tar.gz, we need to unzip it.
    import tarfile
    def un_tar(file_name):
        print("Extracting %s" % file_name)
        tar = tarfile.open(file_name)
        names = tar.getnames()
        # if os.path.isdir(file_name + "_files"):
        #     pass
        # else:
        #     os.mkdir(file_name + "_files")
        for name in names:
            tar.extract(name) #, file_name.split('.')[0])
        tar.close()
        print("Extracted to %s" % names[0])


    if not os.path.exists('cifar-10-batches-py'):
        download(filename)
        un_tar(filename)


    def unpickle(file):
        fp = open(file, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        fp.close()
        return data

    X_train = None
    y_train = []

    path = '' # you can set a dir to the data here.

    for i in range(1,6):
        data_dic = unpickle(path+"cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(path+"cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    y_test = np.array(test_data_dic['labels'])

    if shape == (-1, 3, 32, 32):
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)
        # X_train = np.transpose(X_train, (0, 1, 3, 2))
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

        print('Shape of a training image: X_train[0]',X_train[0].shape)

        plt.ion()       # interactive mode
        count = 1
        for row in range(10):
            for col in range(10):
                a = fig.add_subplot(10, 10, count)
                if shape == (-1, 3, 32, 32):
                    # plt.imshow(X_train[count-1], interpolation='nearest')
                    plt.imshow(np.transpose(X_train[count-1], (1, 2, 0)), interpolation='nearest')
                    # plt.imshow(np.transpose(X_train[count-1], (2, 1, 0)), interpolation='nearest')
                elif shape == (-1, 32, 32, 3):
                    plt.imshow(X_train[count-1], interpolation='nearest')
                    # plt.imshow(np.transpose(X_train[count-1], (1, 0, 2)), interpolation='nearest')
                else:
                    raise Exception("Do not support the given 'shape' to plot the image examples")
                plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                count = count + 1
        plt.draw()      # interactive mode
        plt.pause(3)   # interactive mode

        print("X_train:",X_train.shape)
        print("y_train:",y_train.shape)
        print("X_test:",X_test.shape)
        print("y_test:",y_test.shape)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test

def load_ptb_dataset():
    """PTB dataset is used in many LM papers, including "Empirical Evaluation
    and Combination of Advanced Language Modeling Techniques".

    Code References
    ---------------
    tensorflow.models.rnn.ptb import reader

    Download Links
    ---------------
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    """
    # We first define a download function, supporting both Python 2 and 3.
    filename = 'simple-examples.tgz'
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://www.fit.vutbr.cz/~imikolov/rnnlm/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # After downloading, we need to unzip it.
    import tarfile
    def un_tar(file_name):
        print("Extracting %s" % file_name)
        tar = tarfile.open(file_name)
        names = tar.getnames()
        for name in names:
            tar.extract(name)
        tar.close()
        print("Extracted to /simple-examples")

    if not os.path.exists('simple-examples'):
        download(filename)
        un_tar(filename)

    data_path = os.getcwd() + '/simple-examples/data'
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = build_vocab(read_words(train_path))

    train_data = words_to_word_ids(read_words(train_path), word_to_id)
    valid_data = words_to_word_ids(read_words(valid_path), word_to_id)
    test_data = words_to_word_ids(read_words(test_path), word_to_id)
    vocabulary = len(word_to_id)

    # print(read_words(train_path))     # ... 'according', 'to', 'mr.', '<unk>', '<eos>']
    # print(train_data)                 # ...  214,         5,    23,    1,       2]
    # print(word_to_id)                 # ... 'beyond': 1295, 'anti-nuclear': 9599, 'trouble': 1520, '<eos>': 2 ... }
    # print(vocabulary)                 # 10000
    # exit()
    return train_data, valid_data, test_data, vocabulary

def load_matt_mahoney_text8_dataset():
    """Download a text file from Matt Mahoney's website
    if not present, and make sure it's the right size.
    Extract the first file enclosed in a zip file as a list of words.
    The data can be used for testing Word Embedding.

    Returns
    --------
    word_list : a list
        a list of string (word).
        e.g. [.... 'their', 'families', 'who', 'were', 'expelled', 'from', 'jerusalem', ...]

    Example
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> print('Data size', len(words))
    """
    import zipfile

    url = 'http://mattmahoney.net/dc/'

    def download_matt_mahoney_text8(filename, expected_bytes):
      """Download a text file from Matt Mahoney's website
      if not present, and make sure it's the right size."""
      if not os.path.exists(filename):
        print('Downloading ...')
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
      else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
      return filename

    filename = download_matt_mahoney_text8('text8.zip', 31344016)

    with zipfile.ZipFile(filename) as f:
        word_list = f.read(f.namelist()[0]).split()
    return word_list


## text vectorlization
def read_words(filename):
    """File to list format context.

    Parameters
    ----------
    filename : a string
        a file path (like .txt file),

    Returns
    --------
    The context in a list, split by ' ' by default, and use '<eos>' to represent '\n'.
    e.g. [... 'how', 'useful', 'it', "'s" ... ]

    Code References
    ---------------
    tensorflow.models.rnn.ptb.reader
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def read_analogies_file(eval_file='questions-words.txt', word2id={}):
    """Reads through an analogy question file, return its id format.

    eval_data : a string
        The file name.
    word2id : a dictionary
        Mapping words to unique IDs.
    Returns:
    questions: a [n, 4] numpy array containing the analogy question's
             word ids.
             questions_skipped: questions skipped due to unknown words.

    Example
    -------
    eval_file should be in this format :
    : capital-common-countries
    Athens Greece Baghdad Iraq
    Athens Greece Bangkok Thailand
    Athens Greece Beijing China
    Athens Greece Berlin Germany
    Athens Greece Bern Switzerland
    Athens Greece Cairo Egypt
    Athens Greece Canberra Australia
    Athens Greece Hanoi Vietnam
    Athens Greece Havana Cuba
    ...

    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> data, count, dictionary, reverse_dictionary = \
                tl.files.build_words_dataset(words, vocabulary_size, True)
    >>> analogy_questions = tl.files.read_analogies_file( \
                eval_file='questions-words.txt', word2id=dictionary)
    >>> print(analogy_questions)
    ... [[ 3068  1248  7161  1581]
    ... [ 3068  1248 28683  5642]
    ... [ 3068  1248  3878   486]
    ... ...,
    ... [ 1216  4309 19982 25506]
    ... [ 1216  4309  3194  8650]
    ... [ 1216  4309   140   312]]
    """
    questions = []
    questions_skipped = 0
    with open(eval_file, "rb") as analogy_f:
      for line in analogy_f:
          if line.startswith(b":"):  # Skip comments.
                continue
          words = line.strip().lower().split(b" ")  # lowercase
          ids = [word2id.get(w.strip()) for w in words]
          if None in ids or len(ids) != 4:
              questions_skipped += 1
          else:
              questions.append(np.array(ids))
    print("Eval analogy file: ", eval_file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    analogy_questions = np.array(questions, dtype=np.int32)
    return analogy_questions

def build_vocab(data):
    """Build vocabulary.
        Given the context in list format
        Return the vocabulary, which is a dictionary for word to id.
        e.g. {'campbell': 2587, 'atlantic': 2247, 'aoun': 6746 .... }

    Parameters
    ----------
    data : a list of string
        the context in list format

    Returns
    --------
    word_to_id : a dictionary
        mapping words to unique IDs.
        e.g. {'campbell': 2587, 'atlantic': 2247, 'aoun': 6746 .... }

    Code References
    ---------------
    tensorflow.models.rnn.ptb.reader

    Examples
    --------
    >>> data_path = os.getcwd() + '/simple-examples/data'
    >>> train_path = os.path.join(data_path, "ptb.train.txt")
    >>> word_to_id = build_vocab(read_txt_words(train_path))
    """
    # data = _read_words(filename)
    counter = collections.Counter(data)
    # print('counter', counter)   # dictionary for the occurrence number of each word, e.g. 'banknote': 1, 'photography': 1, 'kia': 1
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # print('count_pairs',count_pairs)  # convert dictionary to list of tuple, e.g. ('ssangyong', 1), ('swapo', 1), ('wachter', 1)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    # print(words)    # list of words
    # print(word_to_id) # dictionary for word to id, e.g. 'campbell': 2587, 'atlantic': 2247, 'aoun': 6746
    return word_to_id

def build_reverse_dictionary(word_to_id):
    """Given a dictionary for converting word to integer id.
    Returns a reverse dictionary for converting a id to word.
    """
    reverse_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return reverse_dictionary

def build_words_dataset(words, vocabulary_size=50000, printable=False):
    """Build the words dictionary and replace rare words with 'UNK' token.
    The most common word has the smallest integer id.

    Parameters
    ----------
    words : a list of string or byte
        The context in list format
    vocabulary_size : an int
        The maximum vocabulary size, limiting the vocabulary size.
        Then the script replaces rare words with 'UNK' token.
    printable : boolen
        Whether to print the read vocabulary size of the given words.

    Returns
    --------
    data : a list of integer
        The context in a list of ids
    count : a list of tuple and list
        count[0] is a list : the number of rare words
        count[1:] are tuples : the number of occurrence of each word
        e.g. [['UNK', 418391], (b'the', 1061396), (b'of', 593677),
                                        (b'and', 416629), (b'one', 411764)]
    dictionary : a dictionary
        word_to_id, mapping words to unique IDs.
    reverse_dictionary : a dictionary
        id_to_word, mapping id to unique word.

    Example
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = \
    ...     tl.files.build_words_dataset(words, vocabulary_size)

    Code References
    -----------------
    tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    """
    import collections
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    if printable:
        print('Real vocabulary size    {}'.format(len(collections.Counter(words).keys())))
        print('Limited vocabulary size {}'.format(vocabulary_size))
    return data, count, dictionary, reverse_dictionary

def words_to_word_ids(data, word_to_id):
    """Given a context (words) in list format and the vocabulary,
    Returns a list of IDs to represent the context.

    Parameters
    ----------
    data : a list of string or byte
        the context in list format
    word_to_id : a dictionary
        mapping words to unique IDs.

    Returns
    --------
    A list of IDs to represent the context.

    Example
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = \
    ...         tl.files.build_words_dataset(words, vocabulary_size, True)
    >>> context = [b'hello', b'how', b'are', b'you']
    >>> ids = tl.files.words_to_word_ids(words, dictionary)
    >>> context = tl.files.word_ids_to_words(ids, reverse_dictionary)
    >>> print(ids)
    ... [6434, 311, 26, 207]
    >>> print(context)
    ... [b'hello', b'how', b'are', b'you']

    Code References
    ---------------
    tensorflow.models.rnn.ptb.reader
    """
    return [word_to_id[word] for word in data]
    # if isinstance(data[0], str):
    #     # print('is a string object')
    #     return [word_to_id[word] for word in data]
    # else:#if isinstance(s, bytes):
    #     # print('is a unicode object')
    #     # print(data[0])
    #     return [word_to_id[str(word)] for word in data]

def word_ids_to_words(data, id_to_word):
    """Given a context (ids) in list format and the vocabulary,
    Returns a list of words to represent the context.

    Parameters
    ----------
    data : a list of integer
        the context in list format
    id_to_word : a dictionary
        mapping id to unique word.

    Returns
    --------
    A list of string or byte to represent the context.

    Examples
    ---------
    see words_to_word_ids
    """
    return [id_to_word[i] for i in data]

def save_vocab(count, name='vocab.txt'):
    """Save the vocabulary to a file so the model can be reloaded.

    Parameters
    ----------
    count : a list of tuple and list
        count[0] is a list : the number of rare words
        count[1:] are tuples : the number of occurrence of each word
        e.g. [['UNK', 418391], (b'the', 1061396), (b'of', 593677),
                                        (b'and', 416629), (b'one', 411764)]

    Examples
    ---------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = \
                tl.files.build_words_dataset(words, vocabulary_size, True)
    >>> tl.files.save_vocab(count, name='vocab_text8.txt')
    ... vocab_text8.txt
    ... UNK 418391
    ... the 1061396
    ... of 593677
    ... and 416629
    ... one 411764
    ... in 372201
    ... a 325873
    ... to 316376
    """
    pwd = os.getcwd()
    vocabulary_size = len(count)
    with open(os.path.join(pwd, name), "w") as f:
        for i in xrange(vocabulary_size):
            f.write("%s %d\n" % (tf.compat.as_text(count[i][0]), count[i][1]))
    print("%d vocab saved to %s in %s" % (vocabulary_size, name, pwd))


## .npz operations
def save_npz(save_list={}, name='model.npz'):
    """Input parameters and the file name, save parameters into .npz file. Use tl.utils.load_npz() to restore.

    Parameters
    ----------
    save_list : a dictionary
        Parameters want to be saved.
    name : a string or None
        The name of the .npz file.

    Examples
    --------
    >>> tl.files.save_npz(network.all_params , name='model_test.npz')
    ... File saved to: model_test.npz
    >>> load_params = tl.files.load_npz(name='model_test.npz')
    ... Loading param0, (784, 800)
    ... Loading param1, (800,)
    ... Loading param2, (800, 800)
    ... Loading param3, (800,)
    ... Loading param4, (800, 10)
    ... Loading param5, (10,)

    References
    ----------
    http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez
    """
    rename_dict = {}
    for k, value in enumerate(save_list):
        rename_dict.update({'param'+str(k) : value.eval()})
    np.savez(name, **rename_dict)
    print('File saved to: %s' % name)

def load_npz(path='', name='model.npz'):
    """Input the file name, restore the parameters from .npz file. Use tl.utils.save_npz() to save parameters.

    Parameters
    ----------
    name : a string or None
        The name of the .npz file.

    Examples
    --------
    >>> tl.files.save_npz(network.all_params , name='model_test.npz')
    ... File saved to: model_test.npz
    >>> load_params = tl.files.load_npz(name='model_test.npz')
    ... Loading param0, (784, 800)
    ... Loading param1, (800,)
    ... Loading param2, (800, 800)
    ... Loading param3, (800,)
    ... Loading param4, (800, 10)
    ... Loading param5, (10,)

    References
    ----------
    http://stackoverflow.com/questions/22315595/saving-dictionary-of-header-information-using-numpy-savez
    """
    d = np.load( path+name )
    params = []
    for key, val in sorted( d.items() ):
        params.append(val)
        print('Loading %s, %s' % (key, str(val.shape)))
    return params


def npz_to_W_pdf(path=None, regx='w1pre_[0-9]+\.(npz)'):
    """Convert the first weight matrix of .npz file to .pdf by using tl.visualize.W().

    Parameters
    ----------
    path : a string or None
        XXX
    regx : a string
        XXX

    Examples
    --------
    ... convert the first weight matrix of w1_pre...npz file to w1_pre...pdf.
    >>> tl.files.npz_to_W_pdf(path='/Users/.../npz_file/', regx='w1pre_[0-9]+\.(npz)')
    """
    file_list = load_file_list(path=path, regx=regx)
    for f in file_list:
        W = load_npz(path, f)[0]
        print("%s --> %s" % (f, f.split('.')[0]+'.pdf'))
        visualize.W(W, second=10, saveable=True, name=f.split('.')[0], fig_idx=2012)


## Help functions
def load_file_list(path=None, regx='\.npz'):
    """Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : a string or None
        XXX
    regx : a string
        XXX

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
    print('Match file list = %s' % return_list)
    print('Number of files = %d' % len(return_list))
    return return_list
