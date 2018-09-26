#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
TensorLayer provides rich layer implementations trailed for
various benchmarks and domain-specific problems. In addition, we also
support transparent access to native TensorFlow parameters.
For example, we provide not only layers for local response normalization, but also
layers that allow user to apply ``tf.nn.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`__.
"""

from .dataset_loaders.celebA_dataset import *
from .dataset_loaders.cifar10_dataset import *
from .dataset_loaders.cyclegan_dataset import *
from .dataset_loaders.flickr_1M_dataset import *
from .dataset_loaders.flickr_25k_dataset import *
from .dataset_loaders.imdb_dataset import *
from .dataset_loaders.matt_mahoney_dataset import *
from .dataset_loaders.mnist_dataset import *
from .dataset_loaders.mnist_fashion_dataset import *
from .dataset_loaders.mpii_dataset import *
from .dataset_loaders.nietzsche_dataset import *
from .dataset_loaders.ptb_dataset import *
from .dataset_loaders.voc_dataset import *
from .dataset_loaders.wmt_en_fr_dataset import *

from .utils import *

__all__ = [
    # Dataset Loaders
    'load_celebA_dataset',
    'load_cifar10_dataset',
    'load_cyclegan_dataset',
    'load_fashion_mnist_dataset',
    'load_flickr1M_dataset',
    'load_flickr25k_dataset',
    'load_imdb_dataset',
    'load_matt_mahoney_text8_dataset',
    'load_mnist_dataset',
    'load_mpii_pose_dataset',
    'load_nietzsche_dataset',
    'load_ptb_dataset',
    'load_voc_dataset',
    'load_wmt_en_fr_dataset',

    # Util Functions
    'assign_params',
    'del_file',
    'del_folder',
    'download_file_from_google_drive',
    'exists_or_mkdir',
    'file_exists',
    'folder_exists',
    'load_and_assign_npz',
    'load_and_assign_npz_dict',
    'load_ckpt',
    'load_cropped_svhn',
    'load_file_list',
    'load_folder_list',
    'load_npy_to_any',
    'load_npz',
    'maybe_download_and_extract',
    'natural_keys',
    'npz_to_W_pdf',
    'read_file',
    'save_any_to_npy',
    'save_ckpt',
    'save_npz',
    'save_npz_dict',
    #'save_graph',
    #'load_graph',
    #'save_graph_and_params',
    #'load_graph_and_params',
]
