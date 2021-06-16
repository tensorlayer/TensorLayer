#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_cost import *
elif BACKEND == 'mindspore':
    from .mindspore_cost import *
elif BACKEND == 'paddle':
    from .paddle_cost import *
else:
    raise NotImplementedError("This backend is not supported")
