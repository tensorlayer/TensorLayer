#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging
from tensorlayer import visualize

from tensorlayer.files.utils import del_file
from tensorlayer.files.utils import folder_exists
from tensorlayer.files.utils import load_file_list
from tensorlayer.files.utils import maybe_download_and_extract
from tensorlayer.files.utils import natural_keys
from tensorlayer.files.utils import read_file

__all__ = ['load_flickr25k_dataset']


def load_flickr25k_dataset(tag='sky', path="data", n_threads=50, printable=False):
    """Load Flickr25K dataset.

    Returns a list of images by a given tag from Flick25k dataset,
    it will download Flickr25k from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`__
    at the first time you use it.

    Parameters
    ------------
    tag : str or None
        What images to return.
            - If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`__.
            - If you want to get all images, set to ``None``.

    path : str
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int
        The number of thread to read image.
    printable : boolean
        Whether to print infomation when reading images, default is ``False``.

    Examples
    -----------
    Get images with tag of sky

    >>> images = tl.files.load_flickr25k_dataset(tag='sky')

    Get all images

    >>> images = tl.files.load_flickr25k_dataset(tag=None, n_threads=100, printable=True)

    """
    path = os.path.join(path, 'flickr25k')

    filename = 'mirflickr25k.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr25k/'

    # download dataset
    if folder_exists(os.path.join(path, "mirflickr")) is False:
        logging.info("[*] Flickr25k is nonexistent in {}".format(path))
        maybe_download_and_extract(filename, path, url, extract=True)
        del_file(os.path.join(path, filename))

    # return images by the given tag.
    # 1. image path list
    folder_imgs = os.path.join(path, "mirflickr")
    path_imgs = load_file_list(path=folder_imgs, regx='\\.jpg', printable=False)
    path_imgs.sort(key=natural_keys)

    # 2. tag path list
    folder_tags = os.path.join(path, "mirflickr", "meta", "tags")
    path_tags = load_file_list(path=folder_tags, regx='\\.txt', printable=False)
    path_tags.sort(key=natural_keys)

    # 3. select images
    if tag is None:
        logging.info("[Flickr25k] reading all images")
    else:
        logging.info("[Flickr25k] reading images with tag: {}".format(tag))
    images_list = []
    for idx, _v in enumerate(path_tags):
        tags = read_file(os.path.join(folder_tags, path_tags[idx])).split('\n')
        # logging.info(idx+1, tags)
        if tag is None or tag in tags:
            images_list.append(path_imgs[idx])

    images = visualize.read_images(images_list, folder_imgs, n_threads=n_threads, printable=printable)
    return images
