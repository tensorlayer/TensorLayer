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

from .tl_logging import *

from .contrib import *

__all__ = [
    # additional modules
    'contrib',

    # tl_logging
    'DEBUG',
    'debug',
    'ERROR',
    'error',
    'FATAL',
    'fatal',
    'INFO',
    'info',
    'WARN',
    'warn',
    'warning',
    'set_verbosity',
    'get_verbosity'
]
