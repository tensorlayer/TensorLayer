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

from .deprecated_deco import deprecated
from .deprecated_alias_deco import deprecated_alias
from .layer_autoregister_deco import layer_autoregister
from .method_deco import private_method
from .method_deco import protected_method
from .layer_deco import force_return_self

__all__ = []
__all__ += deprecated_deco.__all__
__all__ += deprecated_alias_deco.__all__
__all__ += layer_autoregister_deco.__all__
__all__ += method_deco.__all__
__all__ += layer_deco.__all__
