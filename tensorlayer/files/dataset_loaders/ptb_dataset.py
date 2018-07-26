#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import nlp
from tensorlayer import logging

from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_ptb_dataset']


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
    vocab_size = len(word_to_id)

    # logging.info(nlp.read_words(train_path)) # ... 'according', 'to', 'mr.', '<unk>', '<eos>']
    # logging.info(train_data)                 # ...  214,         5,    23,    1,       2]
    # logging.info(word_to_id)                 # ... 'beyond': 1295, 'anti-nuclear': 9599, 'trouble': 1520, '<eos>': 2 ... }
    # logging.info(vocabulary)                 # 10000
    # exit()
    return train_data, valid_data, test_data, vocab_size
