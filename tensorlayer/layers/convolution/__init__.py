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

from .atrous_deconv import *
from .binary_conv import *
from .deformable_conv import *
from .depthwise_conv import *
from .dorefa_conv import *
from .expert_conv import *
from .expert_deconv import *
from .group_conv import *
from .separable_conv import *
from .simplified_conv import *
from .simplified_deconv import *
from .super_resolution import *
from .ternary_conv import *
from .quan_conv import *
from .quan_conv_bn import *

__all__ = []
__all__ += atrous_deconv.__all__
__all__ += binary_conv.__all__
__all__ += deformable_conv.__all__
__all__ += depthwise_conv.__all__
__all__ += dorefa_conv.__all__
__all__ += expert_conv.__all__
__all__ += expert_deconv.__all__
__all__ += group_conv.__all__
__all__ += separable_conv.__all__
__all__ += simplified_conv.__all__
__all__ += simplified_deconv.__all__
__all__ += super_resolution.__all__
__all__ += ternary_conv.__all__
__all__ += quan_conv.__all__
__all__ += quan_conv_bn.__all__
