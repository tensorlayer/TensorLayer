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

from .atrous_conv import *
from .deformable_conv import *
from .depthwise_conv import *
from .expert_conv import *
from .expert_deconv import *
from .group_conv import *
from .separable_conv import *
from .simplified_conv import *
from .simplified_deconv import *

__all__ = [

    # expert conv
    'Conv1d',
    'Conv2d',

    # expert deconv
    'DeConv2d',
    'DeConv3d',

    # simplified conv
    'Conv1dLayer',
    'Conv2dLayer',
    'Conv3dLayer',

    # simplified conv
    'DeConv2dLayer',
    'DeConv3dLayer',

    # deformable
    'DeformableConv2d',

    # atrous
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'AtrousDeConv2dLayer',

    # depthwise
    'DepthwiseConv2d',

    # separable
    'SeparableConv1d',
    'SeparableConv2d',

    # group
    'GroupConv2d',
]
