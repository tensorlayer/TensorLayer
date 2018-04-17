#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""
from __future__ import absolute_import

try:
    import tensorflow
except ImportError:
    install_instr = "Please make sure you install a recent enough version of TensorFlow."
    raise ImportError("__init__.py : Could not import TensorFlow." + install_instr)

from . import activation
from . import cost
from . import files
from . import iterate
from . import layers
from . import models
from . import utils
from . import visualize
from . import prepro
from . import nlp
from . import rein
from . import distributed

# alias
act = activation
vis = visualize

# Use the following formating: (major, minor, patch, prerelease)
VERSION = (1, 8, 5, 'rc1')
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = 'tensorlayer'
__contact_names__ = 'TensorLayer Contributors'
__contact_emails__ = 'hao.dong11@imperial.ac.uk'
__homepage__ = 'http://tensorlayer.readthedocs.io/en/latest/'
__repository_url__ = 'https://github.com/tensorlayer/tensorlayer'
__download_url__ = 'https://github.com/tensorlayer/tensorlayer'
__description__ = 'Reinforcement Learning and Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, supervised learning, unsupervised learning, reinforcement learning, tensorflow'

global_flag = {}
global_dict = {}
