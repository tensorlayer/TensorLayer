#! /usr/bin/python
# -*- coding: utf-8 -*-

import collections
from collections import Counter
import os
import random
import re
import subprocess
import tempfile
import warnings

from six.moves import urllib
from six.moves import xrange

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorlayer as tl
from tensorlayer.lazy_imports import LazyImport

nltk = LazyImport("nltk")

__all__ = [
    'generate_skip_gram_batch',
    'sample',
    'sample_top',
    'SimpleVocabulary',
    'Vocabulary',
    'process_sentence',
    'create_vocab',
    'simple_read_words',
    'read_words',
    'read_analogies_file',
    'build_vocab',
    'build_reverse_dictionary',
    'build_words_dataset',
    'words_to_word_ids',
    'word_ids_to_words',
    'save_vocab',
    'basic_tokenizer',
    'create_vocabulary',
    'initialize_vocabulary',
    'sentence_to_token_ids',
    'data_to_token_ids',
    'moses_multi_bleu',
]


def generate_skip_gram_batch(data, batch_size, num_skips, skip_window, data_index=0):
    """Generate a training batch for the Skip-Gram model.

    See `Word2Vec example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_word2vec_basic.py>`__.

    Parameters
    ----------
    data : list of data
        To present context, usually a list of integers.
    batch_size : int
        Batch size to return.
    num_skips : int
        How many times to reuse an input to generate a label.
    skip_window : int
        How many words to consider left and right.
    data_index : int
        Index of the context location. This code use `data_index` to instead of yield like ``tl.iterate``.

    Returns
    -------
    batch : list of data
        Inputs.
    labels : list of data
        Labels
    data_index : int
        Index of the context location.

    Examples
    --------
    Setting num_skips=2, skip_window=1, use the right and left words.
    In the same way, num_skips=4, skip_window=2 means use the nearby 4 words.

    >>> data = [1,2,3,4,5,6,7,8,9,10,11]
    >>> batch, labels, data_index = tl.nlp.generate_skip_gram_batch(data=data, batch_size=8, num_skips=2, skip_window=1, data_index=0)
    >>> print(batch)
    [2 2 3 3 4 4 5 5]
    >>> print(labels)
    [[3]
    [1]
    [4]
    [2]
    [5]
    [3]
    [4]
    [6]]

    """
    # global data_index   # you can put data_index outside the function, then
    #       modify the global data_index in the function without return it.
    # note: without using yield, this code use data_index to instead.

    if batch_size % num_skips != 0:
        raise Exception("batch_size should be able to be divided by num_skips.")
    if num_skips > 2 * skip_window:
        raise Exception("num_skips <= 2 * skip_window")
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, data_index


def sample(a=None, temperature=1.0):
    """Sample an index from a probability array.

    Parameters
    ----------
    a : list of float
        List of probabilities.
    temperature : float or None
        The higher the more uniform. When a = [0.1, 0.2, 0.7],
            - temperature = 0.7, the distribution will be sharpen [0.05048273,  0.13588945,  0.81362782]
            - temperature = 1.0, the distribution will be the same [0.1,    0.2,    0.7]
            - temperature = 1.5, the distribution will be filtered [0.16008435,  0.25411807,  0.58579758]
            - If None, it will be ``np.argmax(a)``

    Notes
    ------
    - No matter what is the temperature and input list, the sum of all probabilities will be one. Even if input list = [1, 100, 200], the sum of all probabilities will still be one.
    - For large vocabulary size, choice a higher temperature or ``tl.nlp.sample_top`` to avoid error.

    """
    if a is None:
        raise Exception("a : list of float")
    b = np.copy(a)
    try:
        if temperature == 1:
            return np.argmax(np.random.multinomial(1, a, 1))
        if temperature is None:
            return np.argmax(a)
        else:
            a = np.log(a) / temperature
            a = np.exp(a) / np.sum(np.exp(a))
            return np.argmax(np.random.multinomial(1, a, 1))
    except Exception:
        # np.set_printoptions(threshold=np.nan)
        # tl.logging.info(a)
        # tl.logging.info(np.sum(a))
        # tl.logging.info(np.max(a))
        # tl.logging.info(np.min(a))
        # exit()
        message = "For large vocabulary_size, choice a higher temperature\
         to avoid log error. Hint : use ``sample_top``. "

        warnings.warn(message, Warning)
        # tl.logging.info(a)
        # tl.logging.info(b)
        return np.argmax(np.random.multinomial(1, b, 1))


def sample_top(a=None, top_k=10):
    """Sample from ``top_k`` probabilities.

    Parameters
    ----------
    a : list of float
        List of probabilities.
    top_k : int
        Number of candidates to be considered.

    """
    if a is None:
        a = []

    idx = np.argpartition(a, -top_k)[-top_k:]
    probs = a[idx]
    # tl.logging.info("new %f" % probs)
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice
    # old implementation
    # a = np.array(a)
    # idx = np.argsort(a)[::-1]
    # idx = idx[:top_k]
    # # a = a[idx]
    # probs = a[idx]
    # tl.logging.info("prev %f" % probs)
    # # probs = probs / np.sum(probs)
    # # choice = np.random.choice(idx, p=probs)
    # # return choice


# Vector representations of words (Advanced)  UNDOCUMENT
class SimpleVocabulary(object):
    """Simple vocabulary wrapper, see create_vocab().

    Parameters
    ------------
    vocab : dictionary
        A dictionary that maps word to ID.
    unk_id : int
        The ID for 'unknown' word.

    """

    def __init__(self, vocab, unk_id):
        """Initialize the vocabulary."""
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


class Vocabulary(object):
    """Create Vocabulary class from a given vocabulary and its id-word, word-id convert.
    See create_vocab() and ``tutorial_tfrecord3.py``.

    Parameters
    -----------
    vocab_file : str
        The file contains the vocabulary (can be created via ``tl.nlp.create_vocab``), where the words are the first whitespace-separated token on each line (other tokens are ignored) and the word ids are the corresponding line numbers.
    start_word : str
        Special word denoting sentence start.
    end_word : str
        Special word denoting sentence end.
    unk_word : str
        Special word denoting unknown words.

    Attributes
    ------------
    vocab : dictionary
        A dictionary that maps word to ID.
    reverse_vocab : list of int
        A list that maps ID to word.
    start_id : int
        For start ID.
    end_id : int
        For end ID.
    unk_id : int
        For unknown ID.
    pad_id : int
        For Padding ID.

    Examples
    -------------
    The vocab file looks like follow, includes `start_word` , `end_word` ...

    >>> a 969108
    >>> <S> 586368
    >>> </S> 586368
    >>> . 440479
    >>> on 213612
    >>> of 202290
    >>> the 196219
    >>> in 182598
    >>> with 152984
    >>> and 139109
    >>> is 97322

    """

    def __init__(self, vocab_file, start_word="<S>", end_word="</S>", unk_word="<UNK>", pad_word="<PAD>"):
        if not tf.gfile.Exists(vocab_file):
            tl.logging.fatal("Vocab file %s not found." % vocab_file)
        tl.logging.info("Initializing vocabulary from file: %s" % vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        # assert start_word in reverse_vocab
        # assert end_word in reverse_vocab
        if start_word not in reverse_vocab:  # haodong
            reverse_vocab.append(start_word)
        if end_word not in reverse_vocab:
            reverse_vocab.append(end_word)
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        if pad_word not in reverse_vocab:
            reverse_vocab.append(pad_word)

        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tl.logging.info("Vocabulary from %s : %s %s %s" % (vocab_file, start_word, end_word, unk_word))
        tl.logging.info("    vocabulary with %d words (includes start_word, end_word, unk_word)" % len(vocab))
        # tl.logging.info("     vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
        self.pad_id = vocab[pad_word]
        tl.logging.info("      start_id: %d" % self.start_id)
        tl.logging.info("      end_id  : %d" % self.end_id)
        tl.logging.info("      unk_id  : %d" % self.unk_id)
        tl.logging.info("      pad_id  : %d" % self.pad_id)

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]


def process_sentence(sentence, start_word="<S>", end_word="</S>"):
    """Seperate a sentence string into a list of string words, add start_word and end_word,
    see ``create_vocab()`` and ``tutorial_tfrecord3.py``.

    Parameters
    ----------
    sentence : str
        A sentence.
    start_word : str or None
        The start word. If None, no start word will be appended.
    end_word : str or None
        The end word. If None, no end word will be appended.

    Returns
    ---------
    list of str
        A list of strings that separated into words.

    Examples
    -----------
    >>> c = "how are you?"
    >>> c = tl.nlp.process_sentence(c)
    >>> print(c)
    ['<S>', 'how', 'are', 'you', '?', '</S>']

    Notes
    -------
    - You have to install the following package.
    - `Installing NLTK <http://www.nltk.org/install.html>`__
    - `Installing NLTK data <http://www.nltk.org/data.html>`__

    """
    if start_word is not None:
        process_sentence = [start_word]
    else:
        process_sentence = []
    process_sentence.extend(nltk.tokenize.word_tokenize(sentence.lower()))

    if end_word is not None:
        process_sentence.append(end_word)
    return process_sentence


def create_vocab(sentences, word_counts_output_file, min_word_count=1):
    """Creates the vocabulary of word to word_id.

    See ``tutorial_tfrecord3.py``.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Parameters
    ------------
    sentences : list of list of str
        All sentences for creating the vocabulary.
    word_counts_output_file : str
        The file name.
    min_word_count : int
        Minimum number of occurrences for a word.

    Returns
    --------
    :class:`SimpleVocabulary`
        The simple vocabulary object, see :class:`Vocabulary` for more.

    Examples
    --------
    Pre-process sentences

    >>> captions = ["one two , three", "four five five"]
    >>> processed_capts = []
    >>> for c in captions:
    >>>     c = tl.nlp.process_sentence(c, start_word="<S>", end_word="</S>")
    >>>     processed_capts.append(c)
    >>> print(processed_capts)
    ...[['<S>', 'one', 'two', ',', 'three', '</S>'], ['<S>', 'four', 'five', 'five', '</S>']]

    Create vocabulary

    >>> tl.nlp.create_vocab(processed_capts, word_counts_output_file='vocab.txt', min_word_count=1)
    Creating vocabulary.
      Total words: 8
      Words in vocabulary: 8
      Wrote vocabulary file: vocab.txt

    Get vocabulary object

    >>> vocab = tl.nlp.Vocabulary('vocab.txt', start_word="<S>", end_word="</S>", unk_word="<UNK>")
    INFO:tensorflow:Initializing vocabulary from file: vocab.txt
    [TL] Vocabulary from vocab.txt : <S> </S> <UNK>
    vocabulary with 10 words (includes start_word, end_word, unk_word)
        start_id: 2
        end_id: 3
        unk_id: 9
        pad_id: 0

    """
    tl.logging.info("Creating vocabulary.")

    counter = Counter()

    for c in sentences:
        counter.update(c)
        # tl.logging.info('c',c)
    tl.logging.info("    Total words: %d" % len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    word_counts = [("<PAD>", 0)] + word_counts  # 1st id should be reserved for padding
    # tl.logging.info(word_counts)
    tl.logging.info("    Words in vocabulary: %d" % len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    tl.logging.info("    Wrote vocabulary file: %s" % word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = SimpleVocabulary(vocab_dict, unk_id)

    return vocab


# Vector representations of words
def simple_read_words(filename="nietzsche.txt"):
    """Read context from file without any preprocessing.

    Parameters
    ----------
    filename : str
        A file path (like .txt file)

    Returns
    --------
    str
        The context in a string.

    """
    with open(filename, "r") as f:
        words = f.read()
        return words


def read_words(filename="nietzsche.txt", replace=None):
    """Read list format context from a file.

    For customized read_words method, see ``tutorial_generate_text.py``.

    Parameters
    ----------
    filename : str
        a file path.
    replace : list of str
        replace original string by target string.

    Returns
    -------
    list of str
        The context in a list (split using space).
    """
    if replace is None:
        replace = ['\n', '<eos>']

    with tf.gfile.GFile(filename, "r") as f:
        try:  # python 3.4 or older
            context_list = f.read().replace(*replace).split()
        except Exception:  # python 3.5
            f.seek(0)
            replace = [x.encode('utf-8') for x in replace]
            context_list = f.read().replace(*replace).split()
        return context_list


def read_analogies_file(eval_file='questions-words.txt', word2id=None):
    """Reads through an analogy question file, return its id format.

    Parameters
    ----------
    eval_file : str
        The file name.
    word2id : dictionary
        a dictionary that maps word to ID.

    Returns
    --------
    numpy.array
        A ``[n_examples, 4]`` numpy array containing the analogy question's word IDs.

    Examples
    ---------
    The file should be in this format

    >>> : capital-common-countries
    >>> Athens Greece Baghdad Iraq
    >>> Athens Greece Bangkok Thailand
    >>> Athens Greece Beijing China
    >>> Athens Greece Berlin Germany
    >>> Athens Greece Bern Switzerland
    >>> Athens Greece Cairo Egypt
    >>> Athens Greece Canberra Australia
    >>> Athens Greece Hanoi Vietnam
    >>> Athens Greece Havana Cuba

    Get the tokenized analogy question data

    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size, True)
    >>> analogy_questions = tl.nlp.read_analogies_file(eval_file='questions-words.txt', word2id=dictionary)
    >>> print(analogy_questions)
    [[ 3068  1248  7161  1581]
    [ 3068  1248 28683  5642]
    [ 3068  1248  3878   486]
    ...,
    [ 1216  4309 19982 25506]
    [ 1216  4309  3194  8650]
    [ 1216  4309   140   312]]

    """
    if word2id is None:
        word2id = {}

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
    tl.logging.info("Eval analogy file: %s" % eval_file)
    tl.logging.info("Questions: %d", len(questions))
    tl.logging.info("Skipped: %d", questions_skipped)
    analogy_questions = np.array(questions, dtype=np.int32)
    return analogy_questions


def build_vocab(data):
    """Build vocabulary.

    Given the context in list format.
    Return the vocabulary, which is a dictionary for word to id.
    e.g. {'campbell': 2587, 'atlantic': 2247, 'aoun': 6746 .... }

    Parameters
    ----------
    data : list of str
        The context in list format

    Returns
    --------
    dictionary
        that maps word to unique ID. e.g. {'campbell': 2587, 'atlantic': 2247, 'aoun': 6746 .... }

    References
    ---------------
    - `tensorflow.models.rnn.ptb.reader <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/ptb>`_

    Examples
    --------
    >>> data_path = os.getcwd() + '/simple-examples/data'
    >>> train_path = os.path.join(data_path, "ptb.train.txt")
    >>> word_to_id = build_vocab(read_txt_words(train_path))

    """
    # data = _read_words(filename)
    counter = collections.Counter(data)
    # tl.logging.info('counter %s' % counter)   # dictionary for the occurrence number of each word, e.g. 'banknote': 1, 'photography': 1, 'kia': 1
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    # tl.logging.info('count_pairs %s' % count_pairs)  # convert dictionary to list of tuple, e.g. ('ssangyong', 1), ('swapo', 1), ('wachter', 1)
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    # tl.logging.info(words)    # list of words
    # tl.logging.info(word_to_id) # dictionary for word to id, e.g. 'campbell': 2587, 'atlantic': 2247, 'aoun': 6746
    return word_to_id


def build_reverse_dictionary(word_to_id):
    """Given a dictionary that maps word to integer id.
    Returns a reverse dictionary that maps a id to word.

    Parameters
    ----------
    word_to_id : dictionary
        that maps word to ID.

    Returns
    --------
    dictionary
        A dictionary that maps IDs to words.

    """
    reverse_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return reverse_dictionary


def build_words_dataset(words=None, vocabulary_size=50000, printable=True, unk_key='UNK'):
    """Build the words dictionary and replace rare words with 'UNK' token.
    The most common word has the smallest integer id.

    Parameters
    ----------
    words : list of str or byte
        The context in list format. You may need to do preprocessing on the words, such as lower case, remove marks etc.
    vocabulary_size : int
        The maximum vocabulary size, limiting the vocabulary size. Then the script replaces rare words with 'UNK' token.
    printable : boolean
        Whether to print the read vocabulary size of the given words.
    unk_key : str
        Represent the unknown words.

    Returns
    --------
    data : list of int
        The context in a list of ID.
    count : list of tuple and list
        Pair words and IDs.
            - count[0] is a list : the number of rare words
            - count[1:] are tuples : the number of occurrence of each word
            - e.g. [['UNK', 418391], (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764)]
    dictionary : dictionary
        It is `word_to_id` that maps word to ID.
    reverse_dictionary : a dictionary
        It is `id_to_word` that maps ID to word.

    Examples
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size)

    References
    -----------------
    - `tensorflow/examples/tutorials/word2vec/word2vec_basic.py <https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>`__

    """
    if words is None:
        raise Exception("words : list of str or byte")

    count = [[unk_key, -1]]
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
        tl.logging.info('Real vocabulary size    %d' % len(collections.Counter(words).keys()))
        tl.logging.info('Limited vocabulary size {}'.format(vocabulary_size))
    if len(collections.Counter(words).keys()) < vocabulary_size:
        raise Exception(
            "len(collections.Counter(words).keys()) >= vocabulary_size , the limited vocabulary_size must be less than or equal to the read vocabulary_size"
        )
    return data, count, dictionary, reverse_dictionary


def words_to_word_ids(data=None, word_to_id=None, unk_key='UNK'):
    """Convert a list of string (words) to IDs.

    Parameters
    ----------
    data : list of string or byte
        The context in list format
    word_to_id : a dictionary
        that maps word to ID.
    unk_key : str
        Represent the unknown words.

    Returns
    --------
    list of int
        A list of IDs to represent the context.

    Examples
    --------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size, True)
    >>> context = [b'hello', b'how', b'are', b'you']
    >>> ids = tl.nlp.words_to_word_ids(words, dictionary)
    >>> context = tl.nlp.word_ids_to_words(ids, reverse_dictionary)
    >>> print(ids)
    [6434, 311, 26, 207]
    >>> print(context)
    [b'hello', b'how', b'are', b'you']

    References
    ---------------
    - `tensorflow.models.rnn.ptb.reader <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/ptb>`__

    """
    if data is None:
        raise Exception("data : list of string or byte")
    if word_to_id is None:
        raise Exception("word_to_id : a dictionary")
    # if isinstance(data[0], six.string_types):
    #     tl.logging.info(type(data[0]))
    #     # exit()
    #     tl.logging.info(data[0])
    #     tl.logging.info(word_to_id)
    #     return [word_to_id[str(word)] for word in data]
    # else:

    word_ids = []
    for word in data:
        if word_to_id.get(word) is not None:
            word_ids.append(word_to_id[word])
        else:
            word_ids.append(word_to_id[unk_key])
    return word_ids
    # return [word_to_id[word] for word in data]    # this one

    # if isinstance(data[0], str):
    #     # tl.logging.info('is a string object')
    #     return [word_to_id[word] for word in data]
    # else:#if isinstance(s, bytes):
    #     # tl.logging.info('is a unicode object')
    #     # tl.logging.info(data[0])
    #     return [word_to_id[str(word)] f


def word_ids_to_words(data, id_to_word):
    """Convert a list of integer to strings (words).

    Parameters
    ----------
    data : list of int
        The context in list format.
    id_to_word : dictionary
        a dictionary that maps ID to word.

    Returns
    --------
    list of str
        A list of string or byte to represent the context.

    Examples
    ---------
    >>> see ``tl.nlp.words_to_word_ids``

    """
    return [id_to_word[i] for i in data]


def save_vocab(count=None, name='vocab.txt'):
    """Save the vocabulary to a file so the model can be reloaded.

    Parameters
    ----------
    count : a list of tuple and list
        count[0] is a list : the number of rare words,
        count[1:] are tuples : the number of occurrence of each word,
        e.g. [['UNK', 418391], (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764)]

    Examples
    ---------
    >>> words = tl.files.load_matt_mahoney_text8_dataset()
    >>> vocabulary_size = 50000
    >>> data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size, True)
    >>> tl.nlp.save_vocab(count, name='vocab_text8.txt')
    >>> vocab_text8.txt
    UNK 418391
    the 1061396
    of 593677
    and 416629
    one 411764
    in 372201
    a 325873
    to 316376

    """
    if count is None:
        count = []

    pwd = os.getcwd()
    vocabulary_size = len(count)
    with open(os.path.join(pwd, name), "w") as f:
        for i in xrange(vocabulary_size):
            f.write("%s %d\n" % (tf.compat.as_text(count[i][0]), count[i][1]))
    tl.logging.info("%d vocab saved to %s in %s" % (vocabulary_size, name, pwd))


# Functions for translation


def basic_tokenizer(sentence, _WORD_SPLIT=re.compile(b"([.,!?\"':;)(])")):
    """Very basic tokenizer: split the sentence into a list of tokens.

    Parameters
    -----------
    sentence : tensorflow.python.platform.gfile.GFile Object
    _WORD_SPLIT : regular expression for word spliting.


    Examples
    --------
    >>> see create_vocabulary
    >>> from tensorflow.python.platform import gfile
    >>> train_path = "wmt/giga-fren.release2"
    >>> with gfile.GFile(train_path + ".en", mode="rb") as f:
    >>>    for line in f:
    >>>       tokens = tl.nlp.basic_tokenizer(line)
    >>>       tl.logging.info(tokens)
    >>>       exit()
    [b'Changing', b'Lives', b'|', b'Changing', b'Society', b'|', b'How',
      b'It', b'Works', b'|', b'Technology', b'Drives', b'Change', b'Home',
      b'|', b'Concepts', b'|', b'Teachers', b'|', b'Search', b'|', b'Overview',
      b'|', b'Credits', b'|', b'HHCC', b'Web', b'|', b'Reference', b'|',
      b'Feedback', b'Virtual', b'Museum', b'of', b'Canada', b'Home', b'Page']

    References
    ----------
    - Code from ``/tensorflow/models/rnn/translation/data_utils.py``

    """
    words = []
    sentence = tf.compat.as_bytes(sentence)
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(
        vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True,
        _DIGIT_RE=re.compile(br"\d"), _START_VOCAB=None
):
    r"""Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Parameters
    -----------
    vocabulary_path : str
        Path where the vocabulary will be created.
    data_path : str
        Data file that will be used to create vocabulary.
    max_vocabulary_size : int
        Limit on the size of the created vocabulary.
    tokenizer : function
        A function to use to tokenize each data sentence. If None, basic_tokenizer will be used.
    normalize_digits : boolean
        If true, all digits are replaced by `0`.
    _DIGIT_RE : regular expression function
        Default is ``re.compile(br"\d")``.
    _START_VOCAB : list of str
        The pad, go, eos and unk token, default is ``[b"_PAD", b"_GO", b"_EOS", b"_UNK"]``.

    References
    ----------
    - Code from ``/tensorflow/models/rnn/translation/data_utils.py``

    """
    if _START_VOCAB is None:
        _START_VOCAB = [b"_PAD", b"_GO", b"_EOS", b"_UNK"]
    if not gfile.Exists(vocabulary_path):
        tl.logging.info("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    tl.logging.info("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")
    else:
        tl.logging.info("Vocabulary %s from data %s exists" % (vocabulary_path, data_path))


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file, return the `word_to_id` (dictionary)
    and `id_to_word` (list).

    We assume the vocabulary is stored one-item-per-line, so a file will result in a vocabulary {"dog": 0, "cat": 1}, and this function will also return the reversed-vocabulary ["dog", "cat"].

    Parameters
    -----------
    vocabulary_path : str
        Path to the file containing the vocabulary.

    Returns
    --------
    vocab : dictionary
        a dictionary that maps word to ID.
    rev_vocab : list of int
        a list that maps ID to word.

    Examples
    ---------
    >>> Assume 'test' contains
    dog
    cat
    bird
    >>> vocab, rev_vocab = tl.nlp.initialize_vocabulary("test")
    >>> print(vocab)
    >>> {b'cat': 1, b'dog': 0, b'bird': 2}
    >>> print(rev_vocab)
    >>> [b'dog', b'cat', b'bird']

    Raises
    -------
    ValueError : if the provided vocabulary_path does not exist.

    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(
        sentence, vocabulary, tokenizer=None, normalize_digits=True, UNK_ID=3, _DIGIT_RE=re.compile(br"\d")
):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Parameters
    -----------
    sentence : tensorflow.python.platform.gfile.GFile Object
        The sentence in bytes format to convert to token-ids, see ``basic_tokenizer()`` and ``data_to_token_ids()``.
    vocabulary : dictionary
        Mmapping tokens to integers.
    tokenizer : function
        A function to use to tokenize each sentence. If None, ``basic_tokenizer`` will be used.
    normalize_digits : boolean
        If true, all digits are replaced by 0.

    Returns
    --------
    list of int
        The token-ids for the sentence.

    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(
        data_path, target_path, vocabulary_path, tokenizer=None, normalize_digits=True, UNK_ID=3,
        _DIGIT_RE=re.compile(br"\d")
):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Parameters
    -----------
    data_path : str
        Path to the data file in one-sentence-per-line format.
    target_path : str
        Path where the file with token-ids will be created.
    vocabulary_path : str
        Path to the vocabulary file.
    tokenizer : function
        A function to use to tokenize each sentence. If None, ``basic_tokenizer`` will be used.
    normalize_digits : boolean
        If true, all digits are replaced by 0.

    References
    ----------
    - Code from ``/tensorflow/models/rnn/translation/data_utils.py``

    """
    if not gfile.Exists(target_path):
        tl.logging.info("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        tl.logging.info("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(
                        line, vocab, tokenizer, normalize_digits, UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE
                    )
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
    else:
        tl.logging.info("Target path %s exists" % target_path)


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.

    Parameters
    ------------
    hypotheses : numpy.array.string
        A numpy array of strings where each string is a single example.
    references : numpy.array.string
        A numpy array of strings where each string is a single example.
    lowercase : boolean
        If True, pass the "-lc" flag to the multi-bleu script

    Examples
    ---------
    >>> hypotheses = ["a bird is flying on the sky"]
    >>> references = ["two birds are flying on the sky", "a bird is on the top of the tree", "an airplane is on the sky",]
    >>> score = tl.nlp.moses_multi_bleu(hypotheses, references)

    Returns
    --------
    float
        The BLEU score

    References
    ----------
    - `Google/seq2seq/metric/bleu <https://github.com/google/seq2seq>`__

    """
    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    # Get MOSES multi-bleu script
    try:
        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl"
        )
        os.chmod(multi_bleu_path, 0o755)
    except Exception:  # pylint: disable=W0702
        tl.logging.info("Unable to fetch multi-bleu.perl script, using local.")
        metrics_dir = os.path.dirname(os.path.realpath(__file__))
        bin_dir = os.path.abspath(os.path.join(metrics_dir, "..", "..", "bin"))
        multi_bleu_path = os.path.join(bin_dir, "tools/multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                tl.logging.warning("multi-bleu.perl script returned non-zero exit code")
                tl.logging.warning(error.output)
            bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return np.float32(bleu_score)
