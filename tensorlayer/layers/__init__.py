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

from .activation import *
from .convolution import *
from .contrib import *
from .core import *
from .dense import *
from .deprecated import *  # hao dong
from .dropout import *
from .extend import *
from .image_resampling import *
from .importer import *
from .inputs import *
from .lambda_layers import *
from .merge import *
from .noise import *
from .normalization import *
from .padding import *
from .pooling import *
from .quantize_layers import *
from .recurrent import *
from .reshape import *
from .scale import *
from .spatial_transformer import *
from .stack import *
from .time_distribution import *
from .utils import *

__all__ = []
__all__ += activation.__all__
__all__ += contrib.__all__
__all__ += convolution.__all__
__all__ += core.__all__
__all__ += dense.__all__
__all__ += dropout.__all__
__all__ += extend.__all__
__all__ += image_resampling.__all__
__all__ += importer.__all__
__all__ += inputs.__all__
__all__ += lambda_layers.__all__
__all__ += merge.__all__
__all__ += noise.__all__
__all__ += normalization.__all__
__all__ += padding.__all__
__all__ += pooling.__all__
__all__ += quantize_layers.__all__
__all__ += recurrent.__all__
__all__ += reshape.__all__
__all__ += scale.__all__
__all__ += spatial_transformer.__all__
__all__ += stack.__all__
__all__ += time_distribution.__all__
__all__ += utils.__all__
