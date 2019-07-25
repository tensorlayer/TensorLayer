import os
import logging
import cv2

from ..base import Dataset
from ..utils import maybe_download_and_extract

__all__ = ['ILSVRCMeta', 'ILSVRC12', 'ILSVRC12Files']

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"


class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    Metadata is supposed to be found at/will be downloaded to 'path/name/'

    Parameters
    ----------
    path : str
        a folder path
    name : str
        name of the dataset

    Examples
    --------
    >>> meta = ILSVRCMeta(path='data', name='ilsvrc')
    >>> imglist = meta.get_image_list(train_or_val_or_test, dir_structure)

    """

    def __init__(self, path='data', name='ilsvrc'):
        path = os.path.expanduser(path)
        self.path = os.path.join(path, name)
        logging.info("Load or Download {0} > {1}".format(name.upper(), self.path))
        self.filepath = maybe_download_and_extract('ilsvrc_meta', self.path, CAFFE_ILSVRC12_URL, extract=True)
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

    def get_image_list(self, name, dir_structure='original'):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`ILSVRC12.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        assert dir_structure in ['original', 'train']
        add_label_to_fname = (name != 'train' and dir_structure != 'original')
        if add_label_to_fname:
            synset = self.get_synset_1000()

        fname = os.path.join(self.path, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                cls = int(cls)

                if add_label_to_fname:
                    name = os.path.join(synset[cls], name)

                ret.append((name.strip(), cls))
        assert len(ret), fname
        return ret

    # def get_per_pixel_mean(self, size=None):
    #     """
    #     Args:
    #         size (tuple): image size in (h, w). Defaults to (256, 256).
    #     Returns:
    #         np.ndarray: per-pixel mean of shape (h, w, 3 (BGR)) in range [0, 255].
    #     """
    #     if self.caffepb is None:
    #         self.caffepb = get_caffe_pb()
    #     obj = self.caffepb.BlobProto()
    #
    #     mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
    #     with open(mean_file, 'rb') as f:
    #         obj.ParseFromString(f.read())
    #     arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
    #     arr = np.transpose(arr, [1, 2, 0])
    #     if size is not None:
    #         arr = cv2.resize(arr, size[::-1])
    #     return arr

    @staticmethod
    def guess_dir_structure(dir):
        """
        Return the directory structure of "dir".

        Args:
            dir(str): something like '/path/to/imagenet/val'

        Returns:
            either 'train' or 'original'
        """
        subdir = os.listdir(dir)[0]
        # find a subdir starting with 'n'
        if subdir.startswith('n') and \
                os.path.isdir(os.path.join(dir, subdir)):
            dir_structure = 'train'
        else:
            dir_structure = 'original'
        logging.info(
            "[ILSVRC12] Assuming directory {} has '{}' structure.".format(
                dir, dir_structure))
        return dir_structure


class ILSVRC12Files(Dataset):
    """
    Same as :class:`ILSVRC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, path, train_or_val_or_test, meta_dir,
                 dir_structure=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        assert train_or_val_or_test in ['train', 'test', 'val']
        path = os.path.expanduser(path)
        assert os.path.isdir(path)
        self.full_path = os.path.join(path, train_or_val_or_test)
        self.path = train_or_val_or_test
        # assert os.path.isdir(self.full_path)
        # assert os.path.isdir(meta_dir)

        if train_or_val_or_test == 'train':
            dir_structure = 'train'
        elif dir_structure is None:
            dir_structure = ILSVRCMeta.guess_dir_structure(self.full_path)

        meta = ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(train_or_val_or_test, dir_structure)

        # for fname, _ in self.imglist[:10]:
        #     fname = os.path.join(self.full_path, fname)
        #     assert os.path.isfile(fname), fname

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        fname, label = self.imglist[index]
        fname = os.path.join(self.full_path, fname)
        return fname, label

    # def __iter__(self):
    #     idxs = np.arange(len(self.imglist))
    #     if self.shuffle:
    #         self.rng.shuffle(idxs)
    #     for k in idxs:
    #         fname, label = self.imglist[k]
    #         fname = os.path.join(self.full_dir, fname)
    #         yield [fname, label]


class ILSVRC12(ILSVRC12Files):
    """
    Produces uint8 ILSVRC12 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, path, train_or_test, meta_dir,
                 dir_structure=None, shape=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): One of 'original' or 'train'.
                The directory structure for the 'val' directory.
                'original' means the original decompressed directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level directory structure similar to 'dir/train/'.
                By default, it tries to automatically detect the structure.
                You probably do not need to care about this option because 'original' is what people usually have.

        Example:

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
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

        With the downloaded ILSVRC12_img_*.tar, you can use the following
        command to build the above structure:

        .. code-block:: none

            mkdir val && tar xvf ILSVRC12_img_val.tar -C val
            mkdir test && tar xvf ILSVRC12_img_test.tar -C test
            mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'

        When `dir_structure=='train'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                n01440764/
                  ILSVRC2012_val_00000293.JPEG
                  ...
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...
        """
        super(ILSVRC12, self).__init__(
            path, train_or_test, meta_dir, dir_structure)
        self.shape = shape

    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    # def __iter__(self):
    #     for fname, label in super(ILSVRC12, self).__iter__():
    #         im = cv2.imread(fname, cv2.IMREAD_COLOR)
    #         assert im is not None, fname
    #         yield [im, label]

    def __getitem__(self, index):
        fname, label = super(ILSVRC12, self).__getitem__(index)
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        if self.shape is not None:
            img = cv2.resize(img, self.shape)
        return img, label

    # @staticmethod
    # def get_training_bbox(bbox_dir, imglist):
    #     import xml.etree.ElementTree as ET
    #     ret = []
    #
    #     def parse_bbox(fname):
    #         root = ET.parse(fname).getroot()
    #         size = root.find('size').getchildren()
    #         size = map(int, [size[0].text, size[1].text])
    #
    #         box = root.find('object').find('bndbox').getchildren()
    #         box = map(lambda x: float(x.text), box)
    #         return np.asarray(box, dtype='float32')
    #
    #     with timed_operation('Loading Bounding Boxes ...'):
    #         cnt = 0
    #         for k in tqdm.trange(len(imglist)):
    #             fname = imglist[k][0]
    #             fname = fname[:-4] + 'xml'
    #             fname = os.path.join(bbox_dir, fname)
    #             try:
    #                 ret.append(parse_bbox(fname))
    #                 cnt += 1
    #             except Exception:
    #                 ret.append(None)
    #         logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
    #     return ret


# if __name__ == '__main__':
#     meta = ILSVRCMeta()
#     # print(meta.get_synset_words_1000())
#
#     ds = ILSVRC12('/home/wyx/data/fake_ilsvrc/', 'train', shuffle=False)
#     ds.reset_state()
#
#     for k in ds:
#         from IPython import embed
#         embed()
#         break
