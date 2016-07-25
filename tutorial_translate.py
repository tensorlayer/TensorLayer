#! /usr/bin/python
# -*- coding: utf8 -*-


# Copyright 2016 The TensoLayer Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism for multiple buckets.

This class implements a multi-layer recurrent neural network as encoder,
and an attention-based decoder. This is the same as the model described in
this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
or into the seq2seq library for complete model implementation.
This example also allows to use GRU cells in addition to LSTM cells, and
sampled softmax to handle large output vocabulary size. A single-layer
version of this model, but with bi-directional encoder, was presented in
http://arxiv.org/abs/1409.0473
and sampled softmax is described in Section 3 of the following paper.
http://arxiv.org/abs/1412.2007


References
-----------
tensorflow/models/rnn/translate
https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html

Data
----
http://www.statmt.org/wmt10/
"""

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time

def main_test():
    # Download or Load data
    pass


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        main_test()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
