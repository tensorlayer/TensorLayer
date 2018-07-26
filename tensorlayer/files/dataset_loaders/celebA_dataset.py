#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import zipfile

from tensorlayer import logging

from tensorlayer.files.utils import download_file_from_google_drive
from tensorlayer.files.utils import exists_or_mkdir
from tensorlayer.files.utils import load_file_list

__all__ = ['load_celebA_dataset']


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
