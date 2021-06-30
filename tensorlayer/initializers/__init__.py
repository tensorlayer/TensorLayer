#! /usr/bin/python
# -*- coding: utf-8 -*-

# __all__ = [
#     'Initializer', 'Zeros', 'Ones', 'Constant', 'RandomUniform', 'RandomNormal', 'TruncatedNormal',
#     'deconv2d_bilinear_upsampling_initializer', 'He_Normal'
# ]
from .load_initializers_backend import Initializer
from .load_initializers_backend import Zeros
from .load_initializers_backend import Ones
from .load_initializers_backend import Constant
from .load_initializers_backend import RandomUniform
from .load_initializers_backend import RandomNormal
from .load_initializers_backend import TruncatedNormal
from .load_initializers_backend import deconv2d_bilinear_upsampling_initializer
from .load_initializers_backend import HeNormal

# Alias
zeros = Zeros
ones = Ones
constant = Constant
random_uniform = RandomUniform
random_normal = RandomNormal
truncated_normal = TruncatedNormal
he_normal = HeNormal
