#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

from tensorlayer import logging
from tensorlayer import visualize

from tensorlayer.files.utils import del_file
from tensorlayer.files.utils import folder_exists
from tensorlayer.files.utils import load_file_list
from tensorlayer.files.utils import load_folder_list
from tensorlayer.files.utils import maybe_download_and_extract
from tensorlayer.files.utils import read_file

__all__ = ['load_flickr1M_dataset']


def load_flickr1M_dataset(tag='sky', size=10, path="data", n_threads=50, printable=False):
    """Load Flick1M dataset.

    Returns a list of images by a given tag from Flickr1M dataset,
    it will download Flickr1M from `the official website <http://press.liacs.nl/mirflickr/mirdownload.html>`__
    at the first time you use it.

    Parameters
    ------------
    tag : str or None
        What images to return.
            - If you want to get images with tag, use string like 'dog', 'red', see `Flickr Search <https://www.flickr.com/search/>`__.
            - If you want to get all images, set to ``None``.

    size : int
        integer between 1 to 10. 1 means 100k images ... 5 means 500k images, 10 means all 1 million images. Default is 10.
    path : str
        The path that the data is downloaded to, defaults is ``data/flickr25k/``.
    n_threads : int
        The number of thread to read image.
    printable : boolean
        Whether to print infomation when reading images, default is ``False``.

    Examples
    ----------
    Use 200k images

    >>> images = tl.files.load_flickr1M_dataset(tag='zebra', size=2)

    Use 1 Million images

    >>> images = tl.files.load_flickr1M_dataset(tag='zebra')

    """
    import shutil

    path = os.path.join(path, 'flickr1M')
    logging.info("[Flickr1M] using {}% of images = {}".format(size * 10, size * 100000))
    images_zip = [
        'images0.zip', 'images1.zip', 'images2.zip', 'images3.zip', 'images4.zip', 'images5.zip', 'images6.zip',
        'images7.zip', 'images8.zip', 'images9.zip'
    ]
    tag_zip = 'tags.zip'
    url = 'http://press.liacs.nl/mirflickr/mirflickr1m/'

    # download dataset
    for image_zip in images_zip[0:size]:
        image_folder = image_zip.split(".")[0]
        # logging.info(path+"/"+image_folder)
        if folder_exists(os.path.join(path, image_folder)) is False:
            # logging.info(image_zip)
            logging.info("[Flickr1M] {} is missing in {}".format(image_folder, path))
            maybe_download_and_extract(image_zip, path, url, extract=True)
            del_file(os.path.join(path, image_zip))
            # os.system("mv {} {}".format(os.path.join(path, 'images'), os.path.join(path, image_folder)))
            shutil.move(os.path.join(path, 'images'), os.path.join(path, image_folder))
        else:
            logging.info("[Flickr1M] {} exists in {}".format(image_folder, path))

    # download tag
    if folder_exists(os.path.join(path, "tags")) is False:
        logging.info("[Flickr1M] tag files is nonexistent in {}".format(path))
        maybe_download_and_extract(tag_zip, path, url, extract=True)
        del_file(os.path.join(path, tag_zip))
    else:
        logging.info("[Flickr1M] tags exists in {}".format(path))

    # 1. image path list
    images_list = []
    images_folder_list = []
    for i in range(0, size):
        images_folder_list += load_folder_list(path=os.path.join(path, 'images%d' % i))
    images_folder_list.sort(key=lambda s: int(s.split('/')[-1]))  # folder/images/ddd

    for folder in images_folder_list[0:size * 10]:
        tmp = load_file_list(path=folder, regx='\\.jpg', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.jpg
        images_list.extend([os.path.join(folder, x) for x in tmp])

    # 2. tag path list
    tag_list = []
    tag_folder_list = load_folder_list(os.path.join(path, "tags"))

    # tag_folder_list.sort(key=lambda s: int(s.split("/")[-1]))  # folder/images/ddd
    tag_folder_list.sort(key=lambda s: int(os.path.basename(s)))

    for folder in tag_folder_list[0:size * 10]:
        tmp = load_file_list(path=folder, regx='\\.txt', printable=False)
        tmp.sort(key=lambda s: int(s.split('.')[-2]))  # ddd.txt
        tmp = [os.path.join(folder, s) for s in tmp]
        tag_list += tmp

    # 3. select images
    logging.info("[Flickr1M] searching tag: {}".format(tag))
    select_images_list = []
    for idx, _val in enumerate(tag_list):
        tags = read_file(tag_list[idx]).split('\n')
        if tag in tags:
            select_images_list.append(images_list[idx])

    logging.info("[Flickr1M] reading images with tag: {}".format(tag))
    images = visualize.read_images(select_images_list, '', n_threads=n_threads, printable=printable)
    return images
