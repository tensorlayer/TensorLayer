#! /usr/bin/python
# -*- coding: utf-8 -*-

from tensorlayer.backend import BACKEND

if BACKEND == 'mindspore':
    from .core_mindspore import *
elif BACKEND == 'tensorflow':
    from .core_tensorflow import *
elif BACKEND == 'paddle':
    from .core_paddle import *
else:
    raise ("Unsupported backend:", BACKEND)
