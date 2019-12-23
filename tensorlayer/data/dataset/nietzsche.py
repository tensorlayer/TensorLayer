#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['load_nietzsche_dataset', 'NIETZSCHE']

NIETZSCHE_BASE_URL = 'https://s3.amazonaws.com/text-datasets/'
NIETZSCHE_FILENAME = 'nietzsche.txt'


def load_nietzsche_dataset(name='nietzsche', path='raw_data'):
    """
    Load Nietzsche dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/nietzsche/``.

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
    path = os.path.join(path, name)

    filepath = maybe_download_and_extract(NIETZSCHE_FILENAME, path, NIETZSCHE_BASE_URL)

    with open(filepath, "r") as f:
        words = f.read()
        return words


class NIETZSCHE(Dataset):
    """
    Load Nietzsche dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/nietzsche/``.
    """

    def __init__(self, name='nietzsche', path='raw_data'):
        self.words = load_nietzsche_dataset(name=name, path=path)

    def __getitem__(self, index):
        return self.words[index]

    def __len__(self):
        return len(self.words)
