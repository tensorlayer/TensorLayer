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

from .core_utils import *
from .deconv import *
from .list_dict_utils import *
from .merge import *
from .quantization import *
from .recurrent import *
from .reshape import *
from .spatial_transformer import *
from .ternary import *

__all__ = []
__all__ += core_utils.__all__
__all__ += deconv.__all__
__all__ += list_dict_utils.__all__
__all__ += merge.__all__
__all__ += quantization.__all__
__all__ += recurrent.__all__
__all__ += reshape.__all__
__all__ += spatial_transformer.__all__
__all__ += ternary.__all__
