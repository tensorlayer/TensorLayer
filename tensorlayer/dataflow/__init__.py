#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from tensorlayer.backend.ops.load_backend import BACKEND

if BACKEND == 'tensorflow':
    from .tensorflow_data import *
    from .tensorflow_image import *

elif BACKEND == 'mindspore':
    from .mindspore_data import *
    from .mindspore_image import *

elif BACKEND == 'dragon':
    pass

elif BACKEND == 'paddle':
    pass

else:
    raise NotImplementedError("This backend is not supported")

