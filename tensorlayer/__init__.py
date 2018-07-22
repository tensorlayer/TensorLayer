#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

from __future__ import absolute_import

import os

if 'TENSORLAYER_PACKAGE_BUILDING' not in os.environ:

    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with the one of the following commands:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    if tensorflow.__version__ < "1.6.0" and os.environ.get('READTHEDOCS', None) != 'True':
        raise RuntimeError(
            "TensorLayer does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    from tensorlayer import activation
    from tensorlayer import array_ops
    from tensorlayer import cost
    from tensorlayer import db
    from tensorlayer import decorators
    from tensorlayer import distributed
    from tensorlayer import files
    from tensorlayer import initializers
    from tensorlayer import iterate
    from tensorlayer import layers
    from tensorlayer import lazy_imports
    from tensorlayer import logging
    from tensorlayer import models
    from tensorlayer import nlp
    from tensorlayer import optimizers
    from tensorlayer import prepro
    from tensorlayer import rein
    from tensorlayer import utils
    from tensorlayer import visualize

    # alias
    act = activation
    vis = visualize

    alphas = array_ops.alphas
    alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}

# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (1, 9, 0, "")
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
