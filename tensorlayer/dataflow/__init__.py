#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from tensorlayer.backend.ops.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_data import *

elif BACKEND == 'mindspore':
    from .mindspore_data import *

elif BACKEND == 'paddle':
    from .paddle_data import *

elif BACKEND == 'dragon':
    pass

else:
    raise NotImplementedError("This backend is not supported")
