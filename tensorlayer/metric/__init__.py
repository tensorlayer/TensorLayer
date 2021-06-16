#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorlayer.backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_metric import *
elif BACKEND == 'mindspore':
    from .mindspore_metric import *
elif BACKEND == 'paddle':
    from .paddle_metric import *
else:
    raise NotImplementedError("This backend is not supported")
