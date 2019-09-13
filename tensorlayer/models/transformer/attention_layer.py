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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl


class MultiHeadAttentionLayer(tl.layers.Layer):
    """The :class:`MultiHeadAttentionLayer` layer is for multi-head attention computation.
    The weight computation is between "key" and "query", which will then matmul with "value" to generate information
    that selectively focuses on the "query" messages.

    Parameters
    -----------
    num_heads : int
        The number of heads which allow attention computation for different features
    hidden_size : int
        Out dim for the layer
    keep_prob : float
        Keep probablity for drop-out mechanism between 0 and 1
    """

    def __init__(self, num_heads, hidden_size, keep_prob):

        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(hidden_size, num_heads)
            )

        super(MultiHeadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = 1 - keep_prob

        self.build(None)
        self._built = True

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
        }

    def build(self, inputs_shape):
        # Transformation for linearly projecting the queries, keys, and values.
        self.q_transformation = self._get_weights(
            "q_project", shape=(self.hidden_size, self.hidden_size), init=tf.initializers.get('glorot_uniform')
        )
        self.v_transformation = self._get_weights(
            "v_project", shape=(self.hidden_size, self.hidden_size), init=tf.initializers.get('glorot_uniform')
        )
        self.k_transformation = self._get_weights(
            "k_project", shape=(self.hidden_size, self.hidden_size), init=tf.initializers.get('glorot_uniform')
        )
        self.out_transformation = self._get_weights(
            "out_project", shape=(self.hidden_size, self.hidden_size), init=tf.initializers.get('glorot_uniform')
        )

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Parameters
    -----------

      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
    -----------
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
    -----------
      A tensor with shape [batch_size, length, hidden_size]
    """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def forward(self, x, y, mask, cache=None):
        """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      mask: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
    -----------
      Attention layer output with shape [batch_size, length_x, hidden_size]
      Attention weights with shape [batch_size, number_of_head, length_x, length_y]
    """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).

        v = k = y
        q = x

        q = tf.tensordot(q, self.q_transformation, axes=[[2], [0]])
        k = tf.tensordot(k, self.k_transformation, axes=[[2], [0]])
        v = tf.tensordot(v, self.v_transformation, axes=[[2], [0]])

        if cache is not None:

            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)  #(Batch, num_head, length_v, dk)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth**-0.5

        # Calculate dot product attention
        logits = tf.matmul(q, k, transpose_b=True)  #(Batch, num_head, length_q, length_k)
        logits += mask
        weights = tf.nn.softmax(logits, name="attention_weights")  #(Batch, num_head, length_q, length_k)
        weights_store = weights
        if self.is_train:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)

        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = tf.tensordot(attention_output, self.out_transformation, axes=[[2], [0]])
        return attention_output, weights_store


class SelfAttentionLayer(MultiHeadAttentionLayer):
    """Multiheaded self-attention layer."""

    def forward(self, inputs, mask, cache=None):
        return super(SelfAttentionLayer, self).forward(x=inputs, y=inputs, mask=mask, cache=cache)
