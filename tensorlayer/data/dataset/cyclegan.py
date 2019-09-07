#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np

from tensorlayer import logging, visualize

from ..base import Dataset
from ..utils import maybe_download_and_extract, folder_exists, del_file, load_file_list

__all__ = ['load_cyclegan_dataset', 'CycleGAN', 'CycleGANFiles']

CYCLEGAN_BASE_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'


def load_cyclegan_dataset(name='cyclegan', path='raw_data', filename='summer2winter_yosemite'):
    """
    Load images from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.

    Parameters
    ------------
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is `raw_data/cyclegan`.
    filename : str
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.

    Examples
    ---------
    >>> im_train_A, im_train_B, im_test_A, im_test_B = load_cyclegan_dataset(filename='summer2winter_yosemite')

    """
    path = os.path.join(path, name)

    if folder_exists(os.path.join(path, filename)) is False:
        logging.info("[*] {} is nonexistent in {}".format(filename, path))
        filepath = maybe_download_and_extract(filename=filename + '.zip', working_directory=path,
                                              url_source=CYCLEGAN_BASE_URL, extract=True)
        del_file(filepath)

    def load_image_from_folder(path):
        path_imgs = load_file_list(path=path, regx='\\.jpg', printable=False)
        return visualize.read_images(path_imgs, path=path, n_threads=10, printable=False)

    im_train_A = load_image_from_folder(os.path.join(path, filename, "trainA"))
    im_train_B = load_image_from_folder(os.path.join(path, filename, "trainB"))
    im_test_A = load_image_from_folder(os.path.join(path, filename, "testA"))
    im_test_B = load_image_from_folder(os.path.join(path, filename, "testB"))

    def if_2d_to_3d(images):  # [h, w] --> [h, w, 3]
        for i, _v in enumerate(images):
            if len(images[i].shape) == 2:
                images[i] = images[i][:, :, np.newaxis]
                images[i] = np.tile(images[i], (1, 1, 3))
        return images

    im_train_A = if_2d_to_3d(im_train_A)
    im_train_B = if_2d_to_3d(im_train_B)
    im_test_A = if_2d_to_3d(im_test_A)
    im_test_B = if_2d_to_3d(im_test_B)

    return im_train_A, im_train_B, im_test_A, im_test_B


class CycleGANFiles(Dataset):
    """
    Load image file names from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.

    Parameters
    ------------
    train_or_test : str
        Must be either 'train' or 'test'. Choose the training or test dataset.
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is `raw_data/cyclegan`
    filename : str
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.
    """
    def __init__(self, train_or_test, name='cyclegan', path='raw_data', filename='summer2winter_yosemite'):
        self.path = os.path.join(path, name)
        self.train_or_test = train_or_test

        if folder_exists(os.path.join(path, filename)) is False:
            logging.info("[*] {} is nonexistent in {}".format(filename, path))
            filepath = maybe_download_and_extract(filename=filename + '.zip', working_directory=path,
                                                  url_source=CYCLEGAN_BASE_URL, extract=True)
            del_file(filepath)

        assert self.train_or_test in ['train', 'test']
        if self.train_or_test == 'train':
            self.im_A_path = load_file_list(path=os.path.join(path, filename, "trainA"), regx='\\.jpg', printable=False)
            self.im_B_path = load_file_list(path=os.path.join(path, filename, "trainB"), regx='\\.jpg', printable=False)
        else:
            self.im_A_path = load_file_list(path=os.path.join(path, filename, "testA"), regx='\\.jpg', printable=False)
            self.im_B_path = load_file_list(path=os.path.join(path, filename, "testB"), regx='\\.jpg', printable=False)

    def __getitem__(self, index):
        return self.im_A_path[index], self.im_B_path[index]

    def __len__(self):
        assert len(self.im_A_path) == len(self.im_B_path)
        return len(self.im_A_path)


class CycleGAN(CycleGANFiles):
    """
    Load images from CycleGAN's database, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.

    Parameters
    ------------
    train_or_test : str
        Must be either 'train' or 'test'. Choose the training or test dataset.
    name : str
        The name of the dataset.
    path : str
        The path that the data is downloaded to, defaults is `raw_data/cyclegan`
    filename : str
        The dataset you want, see `this link <https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>`.
    """
    def __init__(self, train_or_test, name='cyclegan', path='raw_data', filename='summer2winter_yosemite'):
        super(CycleGAN, self).__init__(train_or_test, name, path, filename)

    def __getitem__(self, index):
        imA = cv2.imread(self.im_A_path)
        imB = cv2.imread(self.im_B_path)
        imA = np.array(imA, dtype=np.float32)
        imB = np.array(imB, dtype=np.float32)
        return imA, imB
