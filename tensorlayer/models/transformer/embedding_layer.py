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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl


class EmbeddingLayer(tl.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        """Specify characteristic parameters of embedding layer.

    Parameters
    -----------
    vocab_size : int
        Number of tokens in the embedding. (Typically ~32,000)
    hidden_size : int
        Dimensionality of the embedding. (Typically 512 or 1024)

    Examples
    ---------
    with TensorLayer

    
    """
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        with tf.name_scope("embedding_and_softmax"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.W = self._get_weights(
                'weights', shape=(self.vocab_size, self.hidden_size),
                init=tf.random_normal_initializer(mean=0., stddev=self.hidden_size**-0.5)
            )

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }

    def forward(self, inputs, mode="embedding"):
        """Get token embeddings of inputs."""
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs):
        """Applies embedding based on inputs tensor."""
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
            embeddings = tf.gather(self.W, inputs)
            embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size**0.5
            return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer."""
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.W, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
