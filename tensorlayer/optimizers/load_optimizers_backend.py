#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayer.backend.ops.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_optimizer import *
elif BACKEND == 'mindspore':
    from .mindspore_optimizer import *
elif BACKEND == 'dragon':
    from .dragon_optimizers import *
else:
    raise NotImplementedError("This backend is not supported")
