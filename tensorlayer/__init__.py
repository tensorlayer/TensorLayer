#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

import os
from distutils.version import LooseVersion

from tensorlayer.package_info import VERSION
from tensorlayer.package_info import __shortversion__
from tensorlayer.package_info import __version__

from tensorlayer.package_info import __package_name__
from tensorlayer.package_info import __contact_names__
from tensorlayer.package_info import __contact_emails__
from tensorlayer.package_info import __homepage__
from tensorlayer.package_info import __repository_url__
from tensorlayer.package_info import __download_url__
from tensorlayer.package_info import __description__
from tensorlayer.package_info import __license__
from tensorlayer.package_info import __keywords__

if 'TENSORLAYER_PACKAGE_BUILDING' not in os.environ:

    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with the one of the following commands:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    if ("SPHINXBUILD" not in os.environ and "READTHEDOCS" not in os.environ and
            LooseVersion(tensorflow.__version__) < LooseVersion("1.6.0")):
        raise RuntimeError(
            "TensorLayer does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

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

    from tensorlayer.lazy_imports import LazyImport

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
