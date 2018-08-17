#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
TensorLayer provides rich layer implementations trailed for
various benchmarks and domain-specific problems. In addition, we also
support transparent access to native TensorFlow parameters.
For example, we provide not only layers for local response normalization, but also
layers that allow user to apply ``tf.nn.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`__.
"""

from .base_pooling import *
from .max_pool import *
from .mean_pool import *
from .global_mean import *
from .global_max import *

__all__ = []
__all__ += base_pooling.__all__
__all__ += max_pool.__all__
__all__ += mean_pool.__all__
__all__ += global_mean.__all__
__all__ += global_max.__all__
