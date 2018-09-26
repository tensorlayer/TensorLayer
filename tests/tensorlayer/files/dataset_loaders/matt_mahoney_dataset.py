#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import zipfile

from tensorlayer import logging

from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_matt_mahoney_text8_dataset']


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
