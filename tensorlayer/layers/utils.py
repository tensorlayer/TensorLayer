#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer import logging

from tensorlayer.decorators import deprecated
from tensorlayer.decorators import deprecated_alias

__all__ = [
    'cabs',
    'clear_layers_name',
    'compute_alpha',
    'flatten_reshape',
    'get_collection_trainable',
    'get_layers_with_name',
    'get_variables_with_name',
    'initialize_global_variables',
    'initialize_rnn_state',
    'list_remove_repeat',
    'merge_networks',
    'print_all_variables',
    '_quantize',
    'quantize_active',
    'quantize_weight',
    'quantize_active_overflow',
    'quantize_weight_overflow',
    'set_name_reuse',
    'ternary_operation',
]

########## Module Public Functions ##########




