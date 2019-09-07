#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import zipfile

import logging

import cv2
import numpy as np

from ..base import Dataset
from ..utils import download_file_from_google_drive, exists_or_mkdir, load_file_list

__all__ = ['load_celebA_dataset', 'CelebAFiles', 'CelebA']


def load_celebA_dataset(name='celebA', path='raw_data'):
    """
    Load CelebA dataset

    Return a list of image path.

    Parameters
    -----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/celebA/``.
    """
    data_dir = name
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


class CelebAFiles(Dataset):
    """
    Load CelebA dataset. Produce filenames of images.

    Parameters
    -----------
    name : str
        The name of the dataset
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/celebA/``.
    """
    def __init__(self, name='celebA', path='raw_data'):
        self.data_files = load_celebA_dataset(name=name, path=path)

    def __getitem__(self, index):
        return self.data_files[index]

    def __len__(self):
        return len(self.data_files)


class CelebA(CelebAFiles):
    """
    Load CelebA dataset. Produce nparrays of images.

    Parameters
    -----------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is ``raw_data/celebA/``.
    shape : tuple
        The shape of digit images.
    """
    def __init__(self, shape=None, name='celebA', path='raw_data'):
        super(CelebA, self).__init__(name=name, path=path)
        self.shape = shape

    def __getitem__(self, index):
        file_path = self.data_files[index]
        img = cv2.imread(file_path)
        if self.shape:
            img = cv2.resize(img, self.shape)
        img = np.array(img, dtype=np.float32)
        return img

    def __len__(self):
        return len(self.data_files)
