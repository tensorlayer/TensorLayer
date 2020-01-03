# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl


class TransformerFeedForwardLayer(tl.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, keep_prob):
        """Initialize FeedForwardNetwork.

    Parameters
    -----------
      hidden_size: int
        output dim of hidden layer.
      filter_size: int
        filter size for the inner (first) dense layer.
      relu_dropout: float
        dropout rate for training.
    """
        super(TransformerFeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = 1 - keep_prob
        self.filter_dense_layer = tl.layers.Dense(
            self.filter_size, in_channels=self.hidden_size, W_init=tf.initializers.get('glorot_uniform'),
            name="input_layer"
        )
        self.output_dense_layer = tl.layers.Dense(
            self.hidden_size, in_channels=self.filter_size, W_init=tf.initializers.get('glorot_uniform'),
            name="output_layer"
        )
        self.build(None)
        self._built = True

    def build(self, inputs_shape):
        pass

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "filter_size": self.filter_size,
            "relu_dropout": self.relu_dropout,
        }

    def forward(self, inputs):
        """Return outputs of the feedforward network."""
        # Retrieve dynamically known shapes
        x = inputs
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        x = tf.reshape(x, [-1, x.shape[-1]])
        output = self.filter_dense_layer(x)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [batch_size, -1, output.shape[-1]])
        if self.is_train:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = tf.reshape(output, [-1, output.shape[-1]])
        output = self.output_dense_layer(output)
        output = tf.reshape(output, [batch_size, -1, output.shape[-1]])

        return output