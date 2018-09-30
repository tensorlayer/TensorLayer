#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.recurrent import DynamicRNNLayer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args


class Seq2Seq(Layer):
    """
    The :class:`Seq2Seq` class is a simple :class:`DynamicRNNLayer` based Seq2seq layer without using `tl.contrib.seq2seq <https://www.tensorflow.org/api_guides/python/contrib.seq2seq>`__.
    See `Model <https://camo.githubusercontent.com/9e88497fcdec5a9c716e0de5bc4b6d1793c6e23f/687474703a2f2f73757269796164656570616e2e6769746875622e696f2f696d672f736571327365712f73657132736571322e706e67>`__
    and `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`__.

    - Please check this example `Chatbot in 200 lines of code <https://github.com/tensorlayer/seq2seq-chatbot>`__.
    - The Author recommends users to read the source code of :class:`DynamicRNNLayer` and :class:`Seq2Seq`.

    Parameters
    ----------
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell
            - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__
            - Note TF1.0+ and TF1.0- are different
    cell_init_args : dictionary or None
        The arguments for the cell initializer.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for the parameters.
    encode_sequence_length : tensor
        For encoder sequence length, see :class:`DynamicRNNLayer` .
    decode_sequence_length : tensor
        For decoder sequence length, see :class:`DynamicRNNLayer` .
    initial_state_encode : None or RNN state
        If None, `initial_state_encode` is zero state, it can be set by placeholder or other RNN.
    initial_state_decode : None or RNN state
        If None, `initial_state_decode` is the final state of the RNN encoder, it can be set by placeholder or other RNN.
    dropout : tuple of float or int
        The input and output keep probability (input_keep_prob, output_keep_prob).
            - If one int, input and output keep probability are the same.
    n_layer : int
        The number of RNN layers, default is 1.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking DenseLayer after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, 2 * n_hidden], for stacking multiple RNN after it.
    name : str
        A unique layer name.

    Attributes
    ------------
    outputs : tensor
        The output of RNN decoder.
    initial_state_encode : tensor or StateTuple
        Initial state of RNN encoder.
    initial_state_decode : tensor or StateTuple
        Initial state of RNN decoder.
    final_state_encode : tensor or StateTuple
        Final state of RNN encoder.
    final_state_decode : tensor or StateTuple
        Final state of RNN decoder.

    Notes
    --------
    - How to feed data: `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/pdf/1409.3215v3.pdf>`__
    - input_seqs : ``['how', 'are', 'you', '<PAD_ID>']``
    - decode_seqs : ``['<START_ID>', 'I', 'am', 'fine', '<PAD_ID>']``
    - target_seqs : ``['I', 'am', 'fine', '<END_ID>', '<PAD_ID>']``
    - target_mask : ``[1, 1, 1, 1, 0]``
    - related functions : tl.prepro <pad_sequences, precess_sequences, sequences_add_start_id, sequences_get_mask>

    Examples
    ----------
    >>> from tensorlayer.layers import *
    >>> batch_size = 32
    >>> encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    >>> decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    >>> target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    >>> target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()
    >>> with tf.variable_scope("model"):
    >>>     # for chatbot, you can use the same embedding layer,
    >>>     # for translation, you may want to use 2 seperated embedding layers
    >>>     with tf.variable_scope("embedding") as vs:
    >>>         net_encode = EmbeddingInputlayer(
    ...                 inputs = encode_seqs,
    ...                 vocabulary_size = 10000,
    ...                 embedding_size = 200,
    ...                 name = 'seq_embedding')
    >>>         vs.reuse_variables()
    >>>         tl.layers.set_name_reuse(True)
    >>>         net_decode = EmbeddingInputlayer(
    ...                 inputs = decode_seqs,
    ...                 vocabulary_size = 10000,
    ...                 embedding_size = 200,
    ...                 name = 'seq_embedding')
    >>>     net = Seq2Seq(net_encode, net_decode,
    ...             cell_fn = tf.contrib.rnn.BasicLSTMCell,
    ...             n_hidden = 200,
    ...             initializer = tf.random_uniform_initializer(-0.1, 0.1),
    ...             encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
    ...             decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
    ...             initial_state_encode = None,
    ...             dropout = None,
    ...             n_layer = 1,
    ...             return_seq_2d = True,
    ...             name = 'seq2seq')
    >>> net_out = DenseLayer(net, n_units=10000, act=None, name='output')
    >>> e_loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')
    >>> y = tf.nn.softmax(net_out.outputs)
    >>> net_out.print_weights(False)

    """

    def __init__(
            self,
            cell_fn,  #tf.nn.rnn_cell.LSTMCell,
            cell_init_args=None,
            n_hidden=256,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            encode_sequence_length=None,
            decode_sequence_length=None,
            initial_state_encode=None,
            initial_state_decode=None,
            dropout=None,
            n_layer=1,
            return_seq_2d=False,
            name='seq2seq',
    ):

        self.cell_fn = cell_fn
        self.cell_init_args = cell_init_args
        self.n_hidden = n_hidden
        self.initializer = initializer
        self.encode_sequence_length = encode_sequence_length
        self.decode_sequence_length = decode_sequence_length
        self.initial_state_encode = initial_state_encode
        self.initial_state_decode = initial_state_decode
        self.dropout = dropout
        self.n_layer = n_layer
        self.return_seq_2d = return_seq_2d
        self.name = name

        if self.cell_init_args:
            self.cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if self.cell_fn is None:
            raise ValueError("cell_fn cannot be set to None")

        if 'GRU' in self.cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")
        # super(Seq2Seq, self).__init__(
        #     prev_layer=[net_encode_in, net_decode_in], cell_init_args=cell_init_args, name=name
        # )

        # logging.info(
        #     "[*] Seq2Seq %s: n_hidden: %d cell_fn: %s dropout: %s n_layer: %d" %
        #     (self.name, n_hidden, cell_fn.__name__, dropout, n_layer)
        # )

        super(Seq2Seq, self).__init__(cell_init_args=cell_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_hidden: %d" % self.n_hidden)
        except AttributeError:
            pass

        try:
            additional_str.append("cell_fn: %s" % self.cell_fn)
        except AttributeError:
            pass

        try:
            additional_str.append("dropout: %s" % self.dropout)
        except AttributeError:
            pass

        try:
            additional_str.append("n_layer: %s" % self.n_layer)
        except AttributeError:
            pass

        try:
            if self.dropout and self._temp_data['is_train']:
                additional_str.append('\n             enable dropout as `is_train` is True')
            elif self.dropout and (self._temp_data['is_train'] is False):
                additional_str.append('\n             disable dropout as `is_train` is True')
        except AttributeError:
            pass

        return self._str(additional_str)

    def __call__(self, net_encode_in, net_decode_in, is_train=True):
        """
        net_encode_in : :class:`Layer`
            Encode sequences, [batch_size, None, n_features].
        net_decode_in : :class:`Layer`
            Decode sequences, [batch_size, None, n_features].
        is_train: boolean (default: True)
            Set the TF Variable in training mode and may impact the behaviour of the layer.
        """

        return super(Seq2Seq, self).__call__(prev_layer=[net_encode_in, net_decode_in], is_train=is_train)

    def compile(self):

        with tf.variable_scope(self.name):

            net_encode_in = self._temp_data['unprocessed_inputs'][0]
            net_decode_in = self._temp_data['unprocessed_inputs'][1]

            # tl.layers.set_name_reuse(reuse)
            # network = InputLayer(self._temp_data['inputs'], name=name+'/input')
            network_encode_layer = DynamicRNNLayer(
                cell_fn=self.cell_fn,
                cell_init_args=self.cell_init_args,
                n_hidden=self.n_hidden,
                initializer=self.initializer,
                initial_state=self.initial_state_encode,
                dropout=self.dropout,
                n_layer=self.n_layer,
                sequence_length=self.encode_sequence_length,
                return_last=False,
                return_seq_2d=True,
                name='encode'
            )

            network_encode_compiled = network_encode_layer(net_encode_in, is_train=self._temp_data['is_train'])

            network_decode_layer = DynamicRNNLayer(
                cell_fn=self.cell_fn,
                cell_init_args=self.cell_init_args,
                n_hidden=self.n_hidden,
                initializer=self.initializer,
                initial_state=(
                    network_encode_compiled.final_state
                    if self.initial_state_decode is None else self.initial_state_decode
                ),
                dropout=self.dropout,
                n_layer=self.n_layer,
                sequence_length=self.decode_sequence_length,
                return_last=False,
                return_seq_2d=self.return_seq_2d,
                name='decode'
            )

            network_decode_compiled = network_decode_layer(net_decode_in, is_train=self._temp_data['is_train'])

            self._temp_data['outputs'] = network_decode_compiled.outputs

            # rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # Initial state
        self._temp_data['initial_state_encode'] = network_encode_compiled.initial_state
        self._temp_data['initial_state_decode'] = network_decode_compiled.initial_state

        # Final state
        self._temp_data['final_state_encode'] = network_encode_compiled.final_state
        self._temp_data['final_state_decode'] = network_decode_compiled.final_state

        self._temp_data['local_weights'] = network_encode_compiled.local_weights + network_decode_compiled.local_weights
