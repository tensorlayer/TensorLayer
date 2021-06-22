#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tensorlayer.backend.ops.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_vision import *
elif BACKEND == 'mindspore':
    from .mindspore_vision import *
elif BACKEND == 'dragon':
    pass
elif BACKEND == 'paddle':
    from .paddle_vision import *
else:
    raise NotImplementedError("This backend is not supported")
