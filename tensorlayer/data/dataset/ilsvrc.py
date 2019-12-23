import numpy as np
import os
import logging
import cv2

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['ILSVRCMeta', 'ILSVRC12', 'ILSVRC12Files']

CAFFE_ILSVRC12_META_BASE_URL = 'http://dl.caffe.berkeleyvision.org/'
CAFFE_ILSVRC12_META_FILENAME = 'caffe_ilsvrc12.tar.gz'


class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    Metadata is supposed to be found at/will be downloaded to 'path/name/'

    Parameters
    ----------
    name : str
        The name of the dataset
    path : str
        The path that the data is downloaded to, defaults is `raw_data/ilsvrc`

    Examples
    --------
    >>> meta = ILSVRCMeta(path='raw_data', name='ilsvrc')
    >>> imglist = meta.get_image_list(train_or_val_or_test, dir_structure)

    """

    def __init__(self, name='ilsvrc', path='raw_data'):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)
        logging.info("Load or Download {0} > {1}".format(name.upper(), self.path))
        self.filepath = maybe_download_and_extract(CAFFE_ILSVRC12_META_FILENAME, self.path, CAFFE_ILSVRC12_META_BASE_URL, extract=True)
        self.caffepb = None

    def get_synset_words_1000(self):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.path, 'synset_words.txt')
        assert os.path.isfile(fname), fname
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.path, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']

        fname = os.path.join(self.path, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                cls = int(cls)
                ret.append((name.strip(), cls))
        assert len(ret), fname
        return ret


class ILSVRC12Files(Dataset):
    """
    Load ILSVRC12 dataset. Produce filenames of images and their corresponding labels.
    Labels are between [0, 999].

    Parameters
    -----------
    train_or_test_or_val : str
        Must be either 'train' or 'test' or 'val'. Choose the training or test or validation dataset.
    meta_dir : str
        The path that the metadata is located. Will automatically download and extract if it is not found.
    path : str
        The path of the ILSVRC12 dataset.


    The dataset should have the structure:
    ---------------------------------------
        path/
            train/
                n02134418/
                    n02134418_198.JPEG
                    ...
                ...
            val/
                ILSVRC2012_val_00000001.JPEG
                ...
            test/
                ILSVRC2012_test_00000001.JPEG
                ...
    ---------------------------------------
    With the downloaded ILSVRC12_img_*.tar, you can use the following
    command to build the above structure:

    mkdir val && tar xvf ILSVRC12_img_val.tar -C val
    mkdir test && tar xvf ILSVRC12_img_test.tar -C test
    mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
    find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
    """
    def __init__(self, train_or_test_or_val, meta_dir, path):
        """
        Same as in :class:`ILSVRC12`.
        """
        assert train_or_test_or_val in ['train', 'test', 'val']
        path = os.path.expanduser(path)
        assert os.path.isdir(path)
        self.full_path = os.path.join(path, train_or_test_or_val)
        self.path = train_or_test_or_val

        meta = ILSVRCMeta(path=meta_dir)
        self.imglist = meta.get_image_list(train_or_test_or_val)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        fname, label = self.imglist[index]
        fname = os.path.join(self.full_path, fname)
        return fname, label


class ILSVRC12(ILSVRC12Files):
    """
    Load ILSVRC12 dataset. Produce images and a label between [0, 999].

    Parameters
    -----------
    train_or_test_or_val : str
        Must be either 'train' or 'test' or 'val'. Choose the training or test or validation dataset.
    meta_dir : str
        The path that the metadata is located. Will automatically download and extract if it is not found.
    path : str
        The path of the ILSVRC12 dataset.
    shape : tuple
        When shape is None, return the original image. If set, return the resized image.


    The dataset should have the structure:
    ---------------------------------------
        path/
            train/
                n02134418/
                    n02134418_198.JPEG
                    ...
                ...
            val/
                ILSVRC2012_val_00000001.JPEG
                ...
            test/
                ILSVRC2012_test_00000001.JPEG
                ...
    ---------------------------------------
    With the downloaded ILSVRC12_img_*.tar, you can use the following
    command to build the above structure:

    mkdir val && tar xvf ILSVRC12_img_val.tar -C val
    mkdir test && tar xvf ILSVRC12_img_test.tar -C test
    mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
    find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
    """
    def __init__(self, train_or_test_or_val, meta_dir, path, shape=None):
        super(ILSVRC12, self).__init__(train_or_test_or_val, meta_dir, path)
        self.shape = shape

    def __getitem__(self, index):
        fname, label = super(ILSVRC12, self).__getitem__(index)
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        if self.shape is not None:
            img = cv2.resize(img, self.shape)
        img = np.array(img, dtype=np.float32)
        return img, label
