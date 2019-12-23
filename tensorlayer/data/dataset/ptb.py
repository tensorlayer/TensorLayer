#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging, nlp
import numpy as np

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['load_ptb_dataset', 'PTB']

PTB_URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/'
PTB_FILENAME = 'simple-examples.tgz'


def load_ptb_dataset(name='ptb', path='raw_data'):
    """
    Load Penn TreeBank (PTB) dataset.

    It is used in many LANGUAGE MODELING papers,
    including "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques", "Recurrent Neural Network Regularization".
    It consists of 929k training words, 73k validation words, and 82k test
    words. It has 10k words in its vocabulary.

    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/ptb/``.

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
    path = os.path.join(path, name)
    logging.info("Load or Download Penn TreeBank (PTB) dataset > {}".format(path))

    # Maybe dowload and uncompress tar, or load exsisting files
    maybe_download_and_extract(PTB_FILENAME, path, PTB_URL, extract=True)

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


class PTB(Dataset):
    """
    Load Penn TreeBank (PTB) dataset.

    It is used in many LANGUAGE MODELING papers,
    including "Empirical Evaluation and Combination of Advanced Language
    Modeling Techniques", "Recurrent Neural Network Regularization".
    It consists of 929k training words, 73k validation words, and 82k test
    words. It has 10k words in its vocabulary.

    Parameters
    ----------
    train_or_test_or_valid : str
        Must be either 'train' or 'test' or 'valid'. Choose the training or test or validation dataset.
    num_steps : int
        The number of unrolls. i.e. sequence_length
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/ptb/``.
    """

    def __init__(self, train_or_test_or_valid, num_steps, name='ptb', path='raw_data'):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)
        logging.info("Load or Download Penn TreeBank (PTB) dataset > {}".format(self.path))

        maybe_download_and_extract(PTB_FILENAME, self.path, PTB_URL, extract=True)

        self.num_steps = num_steps
        self.path = os.path.join(self.path, 'simple-examples', 'data')
        assert train_or_test_or_valid in ['train', 'test', 'valid']
        self.train_or_test_or_valid = train_or_test_or_valid
        train_path = os.path.join(self.path, "ptb.train.txt")
        if train_or_test_or_valid == 'train':
            data_path = train_path
        elif train_or_test_or_valid == 'valid':
            data_path = os.path.join(self.path, "ptb.valid.txt")
        else:
            data_path = os.path.join(self.path, "ptb.test.txt")

        # use training data to build vocab
        self.word_to_id = nlp.build_vocab(nlp.read_words(train_path))
        self.vocav_size = len(self.word_to_id)
        self.data = nlp.words_to_word_ids(nlp.read_words(data_path), self.word_to_id)

        self.data = np.array(self.data, dtype=np.int32)
        self.data_len = (len(self.data) - 1) // self.num_steps
        self.data = self.data[:self.data_len * self.num_steps + 1]

    def __getitem__(self, index):
        x = self.data[index * self.num_steps:(index + 1) * self.num_steps]
        y = self.data[index * self.num_steps + 1:(index + 1) * self.num_steps + 1]
        return x, y

    def __len__(self):
        return self.data_len
