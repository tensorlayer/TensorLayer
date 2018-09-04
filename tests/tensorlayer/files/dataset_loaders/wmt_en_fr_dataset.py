#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import gzip
import tarfile

from tensorflow.python.platform import gfile

from tensorlayer import logging

from tensorlayer.files.utils import maybe_download_and_extract

__all__ = ['load_wmt_en_fr_dataset']


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
