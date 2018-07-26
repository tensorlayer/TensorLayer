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

from .deprecated import deprecated
from .deprecated_alias import deprecated_alias
from .method_decorator import private_method
from .method_decorator import protected_method
from .layer_decorator import force_return_self

__all__ = ['deprecated', 'deprecated_alias', 'private_method', 'protected_method', 'force_return_self']
