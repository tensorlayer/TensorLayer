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
"""Beam search in TF v2.
"""

import tensorflow as tf
import tensorlayer.models.transformer.beamsearchHelper.beam_search_v1 as v1

_StateKeys = v1._StateKeys  # pylint: disable=protected-access


class SequenceBeamSearchV2(v1.SequenceBeamSearch):
    """Implementation of beam search loop in v2."""

    def search(self, initial_ids, initial_cache):
        """Beam search for sequences with highest scores."""
        state, state_shapes = self._create_initial_state(initial_ids, initial_cache)
        finished_state = tf.while_loop(
            self._continue_search, self._search_step, loop_vars=[state], shape_invariants=[state_shapes],
            parallel_iterations=1, back_prop=False
        )
        finished_state = finished_state[0]

        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

        return finished_seq, finished_scores


def sequence_beam_search(
        symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size, alpha, max_decode_length, eos_id
):
    """Search for sequence of subtoken ids with the largest probability.
    
    Parameters
    -----------
    symbols_to_logits_fn : A function with ids, index, and cache as arguments. 
        The passed in arguments will have shape:
            ids -> [batch_size * beam_size, index]
            index -> [] (scalar)
            cache -> nested dictionary of tensors [batch_size * beam_size, ...]
        The function must return logits and new cache.
            logits -> [batch * beam_size, vocab_size]
            new cache -> same shape/structure as inputted cache
    initial_ids : int with shape [batch_size]
        Starting ids for each batch item.
    initial_cache: dict 
        contain starting decoder variables information
    vocab_size: int 
        size of tokens
    beam_size: int 
        number of beams
    alpha: float 
        strength of length normalization
    max_decode_length: int
        maximum length to decoded sequence
    eos_id: int 
        id of eos token, used to determine when a sequence has finished
    
    Notes
    -------
    The function would return:
        Top decoded sequences [batch_size, beam_size, max_decode_length]
        sequence scores [batch_size, beam_size]
  """

    batch_size = tf.shape(initial_ids)[0]

    sbs = SequenceBeamSearchV2(
        symbols_to_logits_fn, vocab_size, batch_size, beam_size, alpha, max_decode_length, eos_id
    )
    return sbs.search(initial_ids, initial_cache)


def _expand_to_same_rank(tensor, target):
    """Expands a given tensor to target's rank to be broadcastable.

    Parameters
    -----------
    
        tensor: input tensor to tile. Shape: [b, d1, ..., da]
        target: target tensor. Shape: [b, d1, ..., da, ..., dn]

     Returns:
    -----------
        Tiled tensor of shape [b, d1, ..., da, 1, ..., 1] with same rank of target.

    Raises:
        ValueError, if the shape rank of rank tensor/target is None.
    """
    if tensor.shape.rank is None:
        raise ValueError("Expect rank for tensor shape, but got None.")
    if target.shape.rank is None:
        raise ValueError("Expect rank for target shape, but got None.")

    with tf.name_scope("expand_rank"):
        diff_rank = target.shape.rank - tensor.shape.rank
        for _ in range(diff_rank):
            tensor = tf.expand_dims(tensor, -1)
        return tensor
