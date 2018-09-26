#! /usr/bin/python
# -*- coding: utf-8 -*-

from .celebA_dataset import *
from .cifar10_dataset import *
from .cyclegan_dataset import *
from .flickr_1M_dataset import *
from .flickr_25k_dataset import *
from .imdb_dataset import *
from .matt_mahoney_dataset import *
from .mnist_dataset import *
from .mnist_fashion_dataset import *
from .mpii_dataset import *
from .nietzsche_dataset import *
from .ptb_dataset import *
from .voc_dataset import *
from .wmt_en_fr_dataset import *

__all__ = [
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
]
