#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

from __future__ import absolute_import

import os
from distutils.version import LooseVersion

if 'TENSORLAYER_PACKAGE_BUILDING' not in os.environ:

    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with the one of the following commands:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    if LooseVersion(tensorflow.__version__) < LooseVersion("1.6.0") and os.environ.get('READTHEDOCS', None) != 'True':
        raise RuntimeError(
            "TensorLayer does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    from . import activation
    from . import array_ops
    from . import cost
    from . import db
    from . import decorators
    from . import distributed
    from . import files
    from . import initializers
    from . import iterate
    from . import layers
    from . import lazy_imports
    from . import tl_logging as logging
    from . import models
    from . import nlp
    from . import optimizers
    from . import prepro
    from . import rein
    from . import utils
    from . import visualize

    # alias
    act = activation
    vis = visualize

    alphas = array_ops.alphas
    alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}

# Use the following formating: (major, minor, patch, prerelease)
VERSION = (1, 9, 1, "")
__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = 'tensorlayer'
__contact_names__ = 'TensorLayer Contributors'
__contact_emails__ = 'tensorlayer@gmail.com'
__homepage__ = 'http://tensorlayer.readthedocs.io/en/latest/'
__repository_url__ = 'https://github.com/tensorlayer/tensorlayer'
__download_url__ = 'https://github.com/tensorlayer/tensorlayer'
__description__ = 'Reinforcement Learning and Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, '
__keywords__ += 'supervised learning, unsupervised learning, reinforcement learning, tensorflow'
