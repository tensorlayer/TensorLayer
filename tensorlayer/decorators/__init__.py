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

from .deprecation_decorators import deprecated
from .deprecation_decorators import deprecated_alias
from .deprecation_decorators import deprecated_args
from .method_decorators import private_method
from .method_decorators import protected_method
from .layer_decorators import force_return_self
from .layer_decorators import layer_autoregister
from .layer_decorators import overwrite_layername_in_network

__all__ = []
__all__ += deprecation_decorators.__all__
__all__ += layer_decorators.__all__
__all__ += method_decorators.__all__
