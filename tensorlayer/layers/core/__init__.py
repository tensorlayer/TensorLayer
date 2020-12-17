#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.backend import BACKEND
if BACKEND == 'mindspore':
    from .core_mindspore import *
elif BACKEND in ['tensorflow', 'dragon']:
    from .core_tensorflow_dragon import *
