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

    from tensorlayer.lazy_imports import LazyImport

    from tensorlayer import activation
    from tensorlayer import array_ops
    from tensorlayer import cost
    from tensorlayer import decorators
    from tensorlayer import files
    from tensorlayer import initializers
    from tensorlayer import iterate
    from tensorlayer import layers
    from tensorlayer import lazy_imports
    from tensorlayer import logging
    from tensorlayer import models
    from tensorlayer import optimizers
    from tensorlayer import rein

    # Lazy Imports
    db = LazyImport("tensorlayer.db")
    distributed = LazyImport("tensorlayer.distributed")
    nlp = LazyImport("tensorlayer.nlp")
    prepro = LazyImport("tensorlayer.prepro")
    utils = LazyImport("tensorlayer.utils")
    visualize = LazyImport("tensorlayer.visualize")

    # alias
    act = activation
    vis = visualize

    alphas = array_ops.alphas
    alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}

# Use the following formatting: (major, minor, patch, prerelease)
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
