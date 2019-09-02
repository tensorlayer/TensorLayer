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
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
import tensorlayer.models.transformer.embedding_layer as embedding_layer
from tensorlayer.models.transformer.attention_layer import SelfAttentionLayer, MultiHeadAttentionLayer
from tensorlayer.models.transformer.feedforward_layer import FeedForwardLayer
from tensorlayer.models.transformer.utils.model_utils import positional_encoding
from tensorlayer.models.transformer.utils.model_utils import get_decoder_self_attention_bias as get_target_mask
from tensorlayer.models.transformer.utils.model_utils import get_padding_bias as get_input_mask
import tensorlayer.models.transformer.beamsearchHelper.beam_search as beam_search


class Transformer(Model):
    """Transformer model.

    Parameters
    ----------
    params: class
        Hyper-parameters of the model including vocab_size, encoder_num_layers, decoder_num_layers, 
        hidden_size, ff_size, num_heads and keep_prob for training; 
        and extra_decode_length, beam_size and alpha for inference.
    
    Examples
    ---------
    >>> class TINY_PARAMS(object):
    >>>     vocab_size = 50
    >>>     encoder_num_layers = 2
    >>>     decoder_num_layers = 2
    >>>     hidden_size = 64
    >>>     ff_size = 16
    >>>     num_heads = 4
    >>>     keep_prob = 0.9
    >>>     extra_decode_length = 5
    >>>     beam_size = 1
    >>>     alpha = 0.6  
    >>> model = Transformer(TINY_PARAMS)

    Returns
    -------
        Stacked-layer transformer model.
    """

    def __init__(self, params, name=None):

        super(Transformer, self).__init__(name=name)
        self.params = params
        self.embedding_softmax_layer = embedding_layer.EmbeddingLayer(params.vocab_size, params.hidden_size)
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)

    def get_config(self):
        return {
            "params": self.params,
        }

    def forward(self, inputs, targets=None):
        """Calculate target logits or inferred target sequences.

    Args:
      inputs: input tensor list of size 1 or 2.
        First item, inputs: int tensor with shape [batch_size, input_length].
        Second item (optional), targets: None or int tensor with shape
          [batch_size, target_length].
      training: boolean, whether in training mode or not.

    Returns:
      If targets is defined:
        Logits for each word in the target sequence: 
            float tensor with shape [batch_size, target_length, vocab_size]
        Self-attention weights for encoder part:
            a dictionary of float tensors {
                "layer_0": [batch_size, number_of_heads, source_length, source_length],
                "layer_1": [batch_size, number_of_heads, source_length, source_length],
                ...
            }
        Weights for decoder part:
            a dictionary of dictionary of float tensors {
                "self": {
                    "layer_0": [batch_size, number_of_heads, target_length, target_length],
                    "layer_1": [batch_size, number_of_heads, target_length, target_length],
                    ...
                }
                "enc_dec": {
                    "layer_0": [batch_size, number_of_heads, source_length, target_length],
                    "layer_1": [batch_size, number_of_heads, source_length, target_length],
                    ...
                }
            }
    
      If target is none:
        Auto-regressive beam-search decoding to generate output each one time step:
            a dictionary {
            outputs: [batch_size, decoded length]
            scores: [batch_size, float]}
            }
        Weights for decoder part:
            a dictionary of dictionary of float tensors {
                "self": {
                    "layer_0": [batch_size, number_of_heads, target_length, target_length],
                    "layer_1": [batch_size, number_of_heads, target_length, target_length],
                    ...
                }
                "enc_dec": {
                    "layer_0": [batch_size, number_of_heads, source_length, target_length],
                    "layer_1": [batch_size, number_of_heads, source_length, target_length],
                    ...
                }
            }
        Self-attention weights for encoder part:
            a dictionary of float tensors {
                "layer_0": [batch_size, number_of_heads, source_length, source_length],
                "layer_1": [batch_size, number_of_heads, source_length, source_length],
                ...
            }

    """
        # # Variance scaling is used here because it seems to work in many problems.
        # # Other reasonable initializers may also work just as well.

        # Calculate attention bias for encoder self-attention and decoder
        # multi-headed attention layers.
        attention_bias = get_input_mask(inputs)

        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        # Prepare inputs to the layer stack by adding positional encodings and
        # applying dropout.
        embedded_inputs = self.embedding_softmax_layer(inputs)
        inputs_padding = get_input_mask(inputs)

        encoder_outputs, weights_encoder = self.encode(inputs, inputs_padding)
        # Generate output sequence if targets is None, or return logits if target
        # sequence is known.
        if targets is None:
            return self.predict(encoder_outputs, attention_bias), weights_encoder
        else:
            logits, weights_decoder = self.decode(targets, encoder_outputs, attention_bias)
        return logits, weights_encoder, weights_decoder

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length].
      training: boolean, whether in training mode or not.

    Returns:
      float tensor with shape [batch_size, input_length, hidden_size]
      Self-attention weights for encoder part:
        a dictionary of float tensors {
            "layer_0": [batch_size, number_of_heads, source_length, source_length],
            "layer_1": [batch_size, number_of_heads, source_length, source_length],
            ...
        }
    """

        # Prepare inputs to the layer stack by adding positional encodings and
        # applying dropout.
        embedded_inputs = self.embedding_softmax_layer(inputs)
        inputs_padding = get_input_mask(inputs)

        length = tf.shape(embedded_inputs)[1]
        pos_encoding = positional_encoding(length, self.params.hidden_size)
        encoder_inputs = embedded_inputs + pos_encoding

        if self.is_train:
            encoder_inputs = tf.nn.dropout(encoder_inputs, rate=1 - self.params.keep_prob)
        return self.encoder_stack(encoder_inputs, input_mask=attention_bias)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence.

    Args:
      targets: target values for the output sequence. int tensor with shape
        [batch_size, target_length]
      encoder_outputs: continuous representation of input sequence. float tensor
        with shape [batch_size, input_length, hidden_size]
      attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
      training: boolean, whether in training mode or not.

    Returns:
      float32 tensor with shape [batch_size, target_length, vocab_size]
      Weights for decoder part:
        a dictionary of dictionary of float tensors {
            "self": {
                "layer_0": [batch_size, number_of_heads, target_length, target_length],
                "layer_1": [batch_size, number_of_heads, target_length, target_length],
                ...
            }
            "enc_dec": {
                "layer_0": [batch_size, number_of_heads, source_length, target_length],
                "layer_1": [batch_size, number_of_heads, source_length, target_length],
                ...
            }
        }
    """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by shifting targets, adding positional
            # encoding and applying dropout.
            decoder_inputs = self.embedding_softmax_layer(targets)
            with tf.name_scope("shift_targets"):
                # Shift targets to the right, and remove the last element
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]
                decoder_inputs += positional_encoding(length, self.params.hidden_size)
            if self.is_train:
                decoder_inputs = tf.nn.dropout(decoder_inputs, rate=1 - self.params.keep_prob)

            # Run values
            decoder_self_attention_bias = get_target_mask(length)
            outputs, weights = self.decoder_stack(
                decoder_inputs,
                features=encoder_outputs,
                input_mask=attention_bias,
                target_mask=decoder_self_attention_bias,
            )
            logits = self.embedding_softmax_layer(outputs, mode="linear")
            return logits, weights

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = positional_encoding(max_decode_length + 1, self.params.hidden_size)
        decoder_self_attention_bias = get_target_mask(max_decode_length)
        weights = []

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs, weight = self.decoder_stack(
                decoder_input, features=cache.get("encoder_outputs"), target_mask=self_attention_bias,
                input_mask=cache.get("encoder_decoder_attention_bias"), cache=cache
            )
            weights.append(weight)
            logits = self.embedding_softmax_layer(decoder_outputs, mode="linear")
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn, weights

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """
    Return predicted sequence, and decoder attention weights.
    """
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn, weights = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)

        # Create cache storing decoder attention values for each layer.
        # pylint: disable=g-complex-comprehension
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.params.hidden_size])
            } for layer in range(self.params.encoder_num_layers)
        }
        # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn, initial_ids=initial_ids, initial_cache=cache,
            vocab_size=self.params.vocab_size, beam_size=self.params.beam_size, alpha=self.params.alpha,
            max_decode_length=max_decode_length, eos_id=1
        )

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        # post-process the weight attention
        for i, weight in enumerate(weights):
            if (i == 0):
                w = weight
            else:
                for k in range(len(w['self'])):
                    w['self']['layer_%d' % k
                             ] = tf.concat([w['self']['layer_%d' % k], weight['self']['layer_%d' % k]], 3)
                    w['enc_dec']['layer_%d' % k
                                ] = tf.concat([w['enc_dec']['layer_%d' % k], weight['enc_dec']['layer_%d' % k]], 2)
        return {"outputs": top_decoded_ids, "scores": top_scores}, w


class LayerNormalization(tl.layers.Layer):
    """
    Layer normalization

    Parameters
    ----------
    hidden_size:
        hidden size of features
    epsilon:
        value to prevent division by zero

    """

    def __init__(self, hidden_size, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.scale = self._get_weights('scale', shape=(self.hidden_size), init=tl.initializers.Ones())
        self.bias = self._get_weights('bias', shape=(self.hidden_size), init=tl.initializers.Zeros())

    def forward(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
        var = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keepdims=True)
        norm_inputs = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return norm_inputs * self.scale + self.bias

    def __repr__(self):
        return "layer normalization"


class PrePostProcessingWrapper(Model):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = 1 - params.keep_prob
        self.layer_norm = LayerNormalization(self.params.hidden_size)

    def get_config(self):
        return {
            "params": self.params,
        }

    def forward(self, inputs, get_weight=False, *args, **kwargs):
        """Calls wrapped layer with same parameters."""

        x = inputs
        y = self.layer_norm(x)

        # Get layer output
        if (get_weight):
            y, weight = self.layer(y, *args, **kwargs)
        else:
            y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.is_train:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        if (get_weight):
            return x + y, weight
        else:
            return x + y


class EncoderStack(Model):
    """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []
        for _ in range(params.encoder_num_layers):
            # Create sublayers for each layer.
            self_attention_layer = SelfAttentionLayer(params.num_heads, params.hidden_size, params.keep_prob)
            feed_forward_network = FeedForwardLayer(params.hidden_size, params.ff_size, params.keep_prob)

            self.layers.append(
                [
                    PrePostProcessingWrapper(self_attention_layer, params),
                    PrePostProcessingWrapper(feed_forward_network, params)
                ]
            )

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params.hidden_size)

    def get_config(self):
        return {
            "params": self.params,
        }

    def forward(self, inputs, input_mask):
        """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Output of encoder layer stack.
      float32 tensor with shape [batch_size, input_length, hidden_size]
    """
        encoder_inputs = inputs
        weights = {}
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope("layer_%d" % n):
                with tf.name_scope("self_attention"):
                    encoder_inputs, weight = self_attention_layer(encoder_inputs, mask=input_mask, get_weight=True)
                    weights["layer_%d" % n] = weight
                with tf.name_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs)

        return self.output_normalization(encoder_inputs), weights


class DecoderStack(Model):
    """Transformer decoder stack.

  Like the encoder stack, the decoder stack is made up of N identical layers.
  Each layer is composed of the sublayers:
    1. Self-attention layer
    2. Multi-headed attention layer combining encoder outputs with results from
       the previous self-attention layer.
    3. Feedforward network (2 fully-connected layers)
  """

    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []
        for _ in range(params.decoder_num_layers):
            self_attention_layer = SelfAttentionLayer(params.num_heads, params.hidden_size, params.keep_prob)
            enc_dec_attention_layer = MultiHeadAttentionLayer(params.num_heads, params.hidden_size, params.keep_prob)
            feed_forward_network = FeedForwardLayer(params.hidden_size, params.ff_size, params.keep_prob)

            self.layers.append(
                [
                    PrePostProcessingWrapper(self_attention_layer, params),
                    PrePostProcessingWrapper(enc_dec_attention_layer, params),
                    PrePostProcessingWrapper(feed_forward_network, params)
                ]
            )
        self.output_normalization = LayerNormalization(params.hidden_size)

    def get_config(self):
        return {
            "params": self.params,
        }

    def forward(self, inputs, features, input_mask, target_mask, cache=None):
        """Return the output of the decoder layer stacks.

    Args:
      decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
      encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: bias for decoder self-attention layer. [1, 1,
        target_len, target_length]
      attention_bias: bias for encoder-decoder attention layer. [batch_size, 1,
        1, input_length]
      training: boolean, whether in training mode or not.
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]},
                       ...}

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
        decoder_inputs = inputs
        decoder_self_attention_bias = target_mask
        encoder_outputs = features
        attention_bias = input_mask
        weights_all = {"self": {}, "enc_dec": {}}
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None

            with tf.name_scope(layer_name):
                with tf.name_scope("self_attention"):
                    decoder_inputs, weight_self = self_attention_layer(
                        decoder_inputs, get_weight=True, mask=decoder_self_attention_bias, cache=layer_cache
                    )
                    weights_all['self']["layer_%d" % n] = weight_self
                with tf.name_scope("encdec_attention"):
                    decoder_inputs, weight_enc_dec = enc_dec_attention_layer(
                        decoder_inputs, get_weight=True, y=encoder_outputs, mask=attention_bias
                    )
                    weights_all['enc_dec']["layer_%d" % n] = weight_enc_dec
                with tf.name_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs), weights_all
