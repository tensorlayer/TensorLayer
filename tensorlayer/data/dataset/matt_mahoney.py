#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import zipfile

from tensorlayer import logging

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['load_matt_mahoney_text8_dataset', 'MattMahoney']

MATT_MAHONEY_BASE_URL = 'http://mattmahoney.net/dc/'
MATT_MAHONEY_FILENAME = 'text8.zip'


def load_matt_mahoney_text8_dataset(name='mm_test8', path='raw_data'):
    """
    Load Matt Mahoney's dataset.

    Download a text file from Matt Mahoney's website
    if not present, and make sure it's the right size.
    Extract the first file enclosed in a zip file as a list of words.
    This dataset can be used for Word Embedding.

    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/mm_test8/``.

    Returns
    --------
    list of str
        The raw text data e.g. [.... 'their', 'families', 'who', 'were', 'expelled', 'from', 'jerusalem', ...]

    Examples
    --------
    >>> words = load_matt_mahoney_text8_dataset()
    >>> print('Data size', len(words))

    """
    path = os.path.join(path, name)
    logging.info("Load or Download matt_mahoney_text8 Dataset> {}".format(path))

    maybe_download_and_extract(MATT_MAHONEY_FILENAME, path, MATT_MAHONEY_BASE_URL, expected_bytes=31344016)

    with zipfile.ZipFile(os.path.join(path, MATT_MAHONEY_FILENAME)) as f:
        word_list = f.read(f.namelist()[0]).split()
        for idx, _ in enumerate(word_list):
            word_list[idx] = word_list[idx].decode()
    return word_list


class MattMahoney(Dataset):
    """
    Load Matt Mahoney's dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/mm_test8/``.
    """

    def __init__(self, name='mm_test8', path='raw_data'):
        self.word_list = load_matt_mahoney_text8_dataset(path=path, name=name)

    def __getitem__(self, index):
        return self.word_list[index]

    def __len__(self):
        return len(self.word_list)
