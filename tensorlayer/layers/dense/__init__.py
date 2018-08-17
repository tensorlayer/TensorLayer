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

from .base_dense import *
from .binary_dense import *
from .dorefa_dense import *
from .dropconnect import *
from .ternary_dense import *
from .quan_dense import *
from .quan_dense_bn import *

__all__ = []
__all__ += base_dense.__all__
__all__ += binary_dense.__all__
__all__ += dorefa_dense.__all__
__all__ += dropconnect.__all__
__all__ += ternary_dense.__all__
__all__ += quan_dense.__all__
__all__ += quan_dense_bn.__all__
