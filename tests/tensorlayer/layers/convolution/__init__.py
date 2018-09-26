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

__all__ = [

    # simplified conv
    'Conv1d',
    'Conv2d',

    # simplified deconv
    'DeConv2d',
    'DeConv3d',

    # expert conv
    'Conv1dLayer',
    'Conv2dLayer',
    'Conv3dLayer',

    # expert conv
    'DeConv2dLayer',
    'DeConv3dLayer',

    # atrous
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'AtrousDeConv2dLayer',

    # binary
    'BinaryConv2d',

    # deformable
    'DeformableConv2d',

    # depthwise
    'DepthwiseConv2d',

    # dorefa
    'DorefaConv2d',

    # group
    'GroupConv2d',

    # separable
    'SeparableConv1d',
    'SeparableConv2d',

    # subpixel
    'SubpixelConv1d',
    'SubpixelConv2d',

    # ternary
    'TernaryConv2d',

    #quan_conv
    'QuanConv2d',
    'QuanConv2dWithBN',
]
