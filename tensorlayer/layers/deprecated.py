# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class KerasLayer(Layer):
    """
    The :class:`KerasLayer` class can be used to merge all Keras layers into
    TensorLayer. Example can be found here `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_.
    This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    keras_layer : function
        A tensor in tensor out function for building model.
    keras_args : dictionary
        The arguments for the `keras_layer`.
    name : str
        A unique layer name.
    """

    def __init__(
            self,
            layer=None,
            keras_layer=None,
            keras_args={},
            name='keras_layer',
    ):
        Layer.__init__(self, name=name)
        assert layer is not None
        assert keras_layer is not None
        self.inputs = layer.outputs
        logging.info("KerasLayer %s: %s" % (self.name, keras_layer))
        logging.info("       This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = keras_layer(self.inputs, **keras_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
        raise Exception("Deprecated : use LambdaLayer instead.")


class EstimatorLayer(Layer):
    """
    The :class:`EstimatorLayer` class accepts ``model_fn`` that described the model.
    It is similar with :class:`KerasLayer`, see `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_.
    This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    model_fn : function
        A tensor in tensor out function for building model.
    args : dictionary
        The arguments for the `model_fn`.
    name : str
        A unique layer name.
    """

    def __init__(
            self,
            layer=None,
            model_fn=None,
            args={},
            name='estimator_layer',
    ):
        Layer.__init__(self, name=name)
        assert layer is not None
        assert model_fn is not None
        self.inputs = layer.outputs
        logging.info("EstimatorLayer %s: %s" % (self.name, model_fn))
        logging.info("       This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = model_fn(self.inputs, **args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
        raise Exception("Deprecated : use LambdaLayer instead.")


class EmbeddingAttentionSeq2seqWrapper(Layer):
    """Sequence-to-sequence model with attention and for multiple buckets (Deprecated after TF0.12).

    This example implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper:
    - `Grammar as a Foreign Language <http://arxiv.org/abs/1412.7449>`_
    please look there for details,
    or into the seq2seq library for complete model implementation.
    This example also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
    - `Neural Machine Translation by Jointly Learning to Align and Translate <http://arxiv.org/abs/1409.0473>`_
    The sampled softmax is described in Section 3 of the following paper.
    - `On Using Very Large Target Vocabulary for Neural Machine Translation <http://arxiv.org/abs/1412.2007>`_

    Parameters
    ----------
    source_vocab_size : size of the source vocabulary.
    target_vocab_size : size of the target vocabulary.
    buckets : a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
    size : number of units in each layer of the model.
    num_layers : number of layers in the model.
    max_gradient_norm : gradients will be clipped to maximally this norm.
    batch_size : the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
    learning_rate : learning rate to start with.
    learning_rate_decay_factor : decay learning rate by this much when needed.
    use_lstm : if true, we use LSTM cells instead of GRU cells.
    num_samples : number of samples for sampled softmax.
    forward_only : if set, we do not construct the backward pass in the model.
    name : str
        A unique layer name
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 use_lstm=False,
                 num_samples=512,
                 forward_only=False,
                 name='wrapper'):
        Layer.__init__(self)  #, name=name)

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if tf.__version__ >= "0.12":
            raise Exception("Deprecated after TF0.12 : use other seq2seq layers instead.")

        # =========== Fake output Layer for compute cost ======
        # If we use sampled softmax, we need an output projection.
        with tf.variable_scope(name) as vs:
            output_projection = None
            softmax_loss_function = None
            # Sampled softmax only makes sense if we sample less than vocabulary size.
            if num_samples > 0 and num_samples < self.target_vocab_size:
                w = tf.get_variable("proj_w", [size, self.target_vocab_size], dtype=D_TYPE)
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=D_TYPE)
                output_projection = (w, b)

                def sampled_loss(inputs, labels):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.target_vocab_size)

                softmax_loss_function = sampled_loss

            # ============ Seq Encode Layer =============
            # Create the internal multi-layer cell for our RNN.
            try:  # TF1.0
                cell_creator = lambda: tf.contrib.rnn.GRUCell(size)
            except:
                cell_creator = lambda: tf.nn.rnn_cell.GRUCell(size)

            if use_lstm:
                try:  # TF1.0
                    cell_creator = lambda: tf.contrib.rnn.BasicLSTMCell(size)
                except:
                    cell_creator = lambda: tf.nn.rnn_cell.BasicLSTMCell(size)

            cell = cell_creator()
            if num_layers > 1:
                try:  # TF1.0
                    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
                except:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            # ============== Seq Decode Layer ============
            # The seq2seq function: we use embedding for the input and attention.
            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                return tf.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode)

            #=============================================================
            # Feeds for inputs.
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
            for i in xrange(buckets[-1][1] + 1):
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

            # Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]
            self.targets = targets  # DH add for debug

            # Training outputs and losses.
            if forward_only:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    targets,
                    self.target_weights,
                    buckets,
                    lambda x, y: seq2seq_f(x, y, True),
                    softmax_loss_function=softmax_loss_function)
                # If we use output projection, we need to project outputs for decoding.
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[b]]
            else:
                self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    targets,
                    self.target_weights,
                    buckets,
                    lambda x, y: seq2seq_f(x, y, False),
                    softmax_loss_function=softmax_loss_function)

            # Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

            # if save into npz
            self.all_params = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # if save into ckpt
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Parameters
        ----------
        session : tensorflow session to use.
        encoder_inputs : list of numpy int vectors to feed as encoder inputs.
        decoder_inputs : list of numpy int vectors to feed as decoder inputs.
        target_weights : list of numpy float vectors to feed as target weights.
        bucket_id : which bucket of the model to use.
        forward_only : whether to do the backward step or only forward.

        Returns
        --------
        A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.

        Raises
        --------
        ValueError : if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket," " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket," " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket," " %d != %d." % (len(target_weights), decoder_size))
        # logging.info('in model.step()')
        # logging.info('a',bucket_id, encoder_size, decoder_size)

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # logging.info(self.encoder_inputs[l].name)
        # logging.info(self.decoder_inputs[l].name)
        # logging.info(self.target_weights[l].name)

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        # logging.info('last_target', last_target)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [
                self.updates[bucket_id],  # Update Op that does SGD.
                self.gradient_norms[bucket_id],  # Gradient norm.
                self.losses[bucket_id]
            ]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id, PAD_ID=0, GO_ID=1, EOS_ID=2, UNK_ID=3):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Parameters
        ----------
        data : a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
        bucket_id : integer, which bucket to get the batch for.
        PAD_ID : int
            Index of Padding in vocabulary
        GO_ID : int
            Index of GO in vocabulary
        EOS_ID : int
            Index of End of sentence in vocabulary
        UNK_ID : int
            Index of Unknown word in vocabulary

        Returns
        -------
        The triple (encoder_inputs, decoder_inputs, target_weights) for
        the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input + [PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][length_idx] for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


# class MaxoutLayer(Layer):
#     """
#     Waiting for contribution
#
#     Single DenseLayer with Max-out behaviour, work well with Dropout.
#
#     References
#     -----------
#     `Goodfellow (2013) Maxout Networks <http://arxiv.org/abs/1302.4389>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         n_units = 100,
#         name ='maxout_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#
#         logging.info("MaxoutLayer %s: %d" % (self.name, self.n_units))
#         logging.info("    Waiting for contribution")
#         with tf.variable_scope(name) as vs:
#             pass
#             # W = tf.Variable(init.xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
#             # b = tf.Variable(tf.zeros([n_units]), name='b')
#
#         # self.outputs = act(tf.matmul(self.inputs, W) + b)
#         # https://www.tensorflow.org/versions/r0.9/api_docs/python/array_ops.html#pack
#         # http://stackoverflow.com/questions/34362193/how-to-explicitly-broadcast-a-tensor-to-match-anothers-shape-in-tensorflow
#         # tf.concat tf.pack  tf.tile
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( [W, b] )

#
