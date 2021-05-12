#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
TensorLayer provides rich layer implementations trailed for
various benchmarks and domain-specific problems. In addition, we also
support transparent access to native TensorFlow parameters.
For example, we provide not only layers for local response normalization, but also
layers that allow user to apply ``tf.ops.lrn`` on ``network.outputs``.
More functions can be found in `TensorFlow API <https://www.tensorflow.org/versions/master/api_docs/index.html>`__.
"""

from .amsgrad import AMSGrad

# ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']
from .load_optimizers_backend import Adadelta
from .load_optimizers_backend import Adagrad
from .load_optimizers_backend import Adam
from .load_optimizers_backend import Adamax
from .load_optimizers_backend import Ftrl
from .load_optimizers_backend import Nadam
from .load_optimizers_backend import RMSprop
from .load_optimizers_backend import SGD
from .load_optimizers_backend import Momentum
from .load_optimizers_backend import Lamb
from .load_optimizers_backend import LARS
