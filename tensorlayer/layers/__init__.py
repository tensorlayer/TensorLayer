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
from .core import *
from .dense import *
from .dropout import *
from .extend import *
from .flow_control import *
from .image_resampling import *
from .importer import *
from .inputs import *
from .lambda_layers import *
from .merge import *
from .noise import *
from .normalization import *
from .object_detection import *
from .padding import *
from .pooling import *
from .quantize import *
from .reconstruction import *
from .recurrent import *
from .scale import *
from .shape import *
from .spatial_transformer import *
from .stack import *
from .time_distribution import *
from .utils import *
