#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers."""

MAJOR = 2
MINOR = 0
PATCH = 1
PRE_RELEASE = ''
# Use the following formatting: (major, minor, patch, prerelease)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'tensorlayer'
__contact_names__ = 'TensorLayer Contributors'
__contact_emails__ = 'tensorlayer@gmail.com'
__homepage__ = 'http://tensorlayer.readthedocs.io/en/latest/'
__repository_url__ = 'https://github.com/tensorlayer/tensorlayer'
__download_url__ = 'https://github.com/tensorlayer/tensorlayer'
__description__ = 'High Level Tensorflow Deep Learning Library for Researcher and Engineer.'
__license__ = 'apache'
__keywords__ = 'deep learning, machine learning, computer vision, nlp, '
__keywords__ += 'supervised learning, unsupervised learning, reinforcement learning, tensorflow'
