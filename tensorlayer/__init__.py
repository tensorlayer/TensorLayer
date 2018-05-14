#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""
from __future__ import absolute_import

try:
    import tensorflow

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

    global_flag = {}
    global_dict = {}

except Exception as e:

    import pkg_resources
    installed_packages = [d for d in pkg_resources.working_set]

    for package in installed_packages:
        if 'tensorlayer' in package.project_name and 'site-packages' in package.location:
            raise ImportError("__init__.py : Could not import TensorLayer.\nError: {}".format(e))

# Use the following formating: (major, minor, patch, prerelease)
VERSION = (1, 8, 5)
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
