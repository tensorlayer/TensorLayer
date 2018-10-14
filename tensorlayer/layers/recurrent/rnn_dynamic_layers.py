#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

from tensorlayer.layers.utils import advanced_indexing_op
from tensorlayer.layers.utils import retrieve_seq_length_op

from tensorlayer import logging

__all__ = [
    'DynamicRNN',
    'BiDynamicRNN',
]


class DynamicRNN(Layer):
    """
    The :class:`DynamicRNN` class is a dynamic recurrent layer, see ``tf.nn.dynamic_rnn``.

    Parameters
    ----------
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell
            - See `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__
            - Note TF1.0+ and TF1.0- are different
    cell_init_args : dictionary or None
        The arguments for the cell function.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : tensor, array or None
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic RNN in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__.
    initial_state : None or RNN State
        If None, `initial_state` is zero state.
    dropout : float
        The input and output keep probability (input_keep_prob, output_keep_prob).
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean or None
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking Dense after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    dynamic_rnn_init_args : dictionary
        The arguments for ``tf.nn.dynamic_rnn``.
    name : str
        A unique layer name.

    Attributes
    ------------
    outputs : tensor
        The output of this layer.

    final_state : tensor or StateTuple
        The finial state of this layer.
            - When `state_is_tuple` is `False`, it is the final hidden and cell states, `states.get_shape() = [?, 2 * n_hidden]`.
            - When `state_is_tuple` is `True`, it stores two elements: `(c, h)`.
            - In practice, you can get the final state after each iteration during training, then feed it to the initial state of next iteration.

    initial_state : tensor or StateTuple
        The initial state of this layer.
            - In practice, you can set your state at the begining of each epoch or iteration according to your training procedure.

    batch_size : int or tensor
        It is an integer, if it is able to compute the `batch_size`; otherwise, tensor for dynamic batch size.

    sequence_length : a tensor or array
        The sequence lengths computed by Advanced Opt or the given sequence lengths, [batch_size]

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`Reshape`.

    Examples
    --------
    Synced sequence input and output, for loss function see ``tl.cost.cross_entropy_seq_with_mask``.

    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
    >>> net = tl.layers.EmbeddingInput(
    ...             vocabulary_size=vocab_size,
    ...             embedding_size=embedding_size,
    ...             name='embedding')(input_seqs)
    >>> net = tl.layers.DynamicRNN(
    ...             cell_fn=tf.contrib.rnn.BasicLSTMCell, # for TF0.2 use tf.nn.rnn_cell.BasicLSTMCell,
    ...             n_hidden=embedding_size,
    ...             dropout=(0.7 if is_train else None),
    ...             sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    ...             return_last=False,                    # for encoder, set to True
    ...             return_seq_2d=True,                   # stack Dense or compute cost after it
    ...             name='dynamicrnn')(net)
    >>> net = tl.layers.Dense(n_units=vocab_size, name="output")(net)

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__
    - `dynamic_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb>`__
    - `tf.nn.dynamic_rnn <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md>`__
    - `tflearn rnn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__
    - ``tutorial_dynamic_rnn.py``

    """

    def __init__(
        self,
        cell_fn=None,
        cell_init_args=None,
        n_hidden=256,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length=None,
        initial_state=None,
        dropout=None,
        n_layer=1,
        return_last=False,
        return_seq_2d=False,
        dynamic_rnn_init_args=None,
        name='dyrnn',
    ):
        if cell_fn is None:
            cell_fn = tf.contrib.rnn.BasicLSTMCell

        if cell_init_args and ('LSTM' in cell_fn.__name__):
            cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        self.cell_fn = cell_fn
        self.cell_init_args = cell_init_args
        self.n_hidden = n_hidden
        self.initializer = initializer
        self.sequence_length = sequence_length
        self.initial_state = initial_state
        self.dropout = dropout
        self.n_layer = n_layer
        self.return_last = return_last
        self.return_seq_2d = return_seq_2d
        self.dynamic_rnn_init_args = dynamic_rnn_init_args
        self.name = name

        super(DynamicRNN, self).__init__(
            cell_init_args=cell_init_args,
            dynamic_rnn_init_args=dynamic_rnn_init_args,
        )

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_hidden: %d" % self.n_hidden)
        except AttributeError:
            pass

        try:
            additional_str.append("cell_fn: %s" % self.cell_fn.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("n_layer: %d" % self.n_layer)
        except AttributeError:
            pass

        try:
            additional_str.append("return_last: %s" % str(self.return_last))
        except AttributeError:
            pass

        try:
            additional_str.append("return_seq_2d: %s" % str(self.return_seq_2d))
        except AttributeError:
            pass

        try:
            additional_str.append("input shape: %s" % self._temp_data['inputs'].get_shape())
        except AttributeError:
            pass

        try:
            _dropout = str(self.dropout) if self._temp_data['is_train'] and self.dropout is not None else "disabled"
            additional_str.append("dropout: %s" % _dropout)
        except AttributeError:
            pass

        return self._str(additional_str)
        # logging.info(
        #     "DynamicRNN %s: n_hidden: %d, in_dim: %d in_shape: %s cell_fn: %s dropout: %s n_layer: %d" % (
        #         self.name, n_hidden, self._temp_data['inputs'].get_shape().ndims, self._temp_data['inputs'].get_shape(),
        #         cell_fn.__name__, dropout, n_layer
        #     )
        # )

    def build(self):

        self._temp_data['dropout'] = self.dropout if self._temp_data['is_train'] else None

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self._temp_data['inputs'].get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self._temp_data['inputs'].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self._temp_data['inputs'])[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        # self.batch_size = batch_size

        # Creats the cell function
        # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **self.cell_init_args) # HanSheng
        rnn_creator = lambda: self.cell_fn(num_units=self.n_hidden, **self.cell_init_args)

        # Apply dropout
        if self._temp_data['dropout'] is not None:
            if isinstance(self._temp_data['dropout'], (tuple, list)):
                in_keep_prob = self._temp_data['dropout'][0]
                out_keep_prob = self._temp_data['dropout'][1]

            elif isinstance(self._temp_data['dropout'], float):
                in_keep_prob, out_keep_prob = self._temp_data['dropout'], self._temp_data['dropout']

            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of " "float)")

            DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper

            # cell_instance_fn1=cell_instance_fn        # HanSheng
            # cell_instance_fn=DropoutWrapper_fn(
            #                     cell_instance_fn1(),
            #                     input_keep_prob=in_keep_prob,
            #                     output_keep_prob=out_keep_prob)
            cell_creator = lambda is_last=True: DropoutWrapper_fn(
                rnn_creator(), input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob if is_last else 1.0
            )
        else:
            cell_creator = rnn_creator
        self.cell = cell_creator()
        # Apply multiple layers
        if self.n_layer > 1:
            try:
                MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell
            except Exception:
                MultiRNNCell_fn = tf.nn.rnn_cell.MultiRNNCell

            # cell_instance_fn2=cell_instance_fn # HanSheng
            if self._temp_data['dropout'] is not None:
                try:
                    # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)], state_is_tuple=True) # HanSheng
                    self.cell = MultiRNNCell_fn(
                        [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)], state_is_tuple=True
                    )
                except Exception:  # when GRU
                    # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)]) # HanSheng
                    self.cell = MultiRNNCell_fn(
                        [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)]
                    )
            else:
                try:
                    self.cell = MultiRNNCell_fn([cell_creator() for _ in range(self.n_layer)], state_is_tuple=True)
                except Exception:  # when GRU
                    self.cell = MultiRNNCell_fn([cell_creator() for _ in range(self.n_layer)])

        # self.cell=cell_instance_fn() # HanSheng

        # Initialize initial_state
        if self.initial_state is None:
            self.initial_state = self.cell.zero_state(
                batch_size, dtype=self._temp_data['inputs'].dtype
            )  # dtype=tf.float32)
        else:
            self.initial_state = self.initial_state

        # Computes sequence_length
        if self.sequence_length is None:

            sequence_length = retrieve_seq_length_op(
                self._temp_data['inputs'] if isinstance(self._temp_data['inputs'], tf.Tensor) else tf.
                stack(self._temp_data['inputs'])
            )

        # Main - Computes outputs and last_states
        with tf.variable_scope(self.name, initializer=self.initializer) as vs:
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=self.cell,
                # inputs=X
                inputs=self._temp_data['inputs'],
                # dtype=tf.float64,
                sequence_length=self.sequence_length,
                initial_state=self.initial_state,
                **self.dynamic_rnn_init_args
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            # logging.info("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            if self.return_last:
                # [batch_size, n_hidden]
                # outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
                self._temp_data['outputs'] = advanced_indexing_op(outputs, sequence_length)

            else:
                # [batch_size, n_step(max), n_hidden]
                # self._temp_data['outputs'] = result[0]["outputs"]
                # self._temp_data['outputs'] = outputs    # it is 3d, but it is a list
                if self.return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, n_hidden]
                    self._temp_data['outputs'] = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])

                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), n_hidden]
                    max_length = tf.shape(outputs)[1]

                    self._temp_data['outputs'] = tf.reshape(
                        tf.concat(outputs, 1),
                        [batch_size, max_length, self.n_hidden]
                    )
                    # self._temp_data['outputs'] = tf.reshape(tf.concat(1, outputs), [-1, max_length, n_hidden])

            self._temp_data['local_weights'] = rnn_variables

        self._temp_data['initial_state'] = self.initial_state
        self._temp_data['final_state'] = last_states
        self._temp_data['sequence_length'] = self.sequence_length


class BiDynamicRNN(Layer):
    """
    The :class:`BiDynamicRNN` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell
            - See `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__.
            - Note TF1.0+ and TF1.0- are different.
    cell_init_args : dictionary
        The arguments for the cell initializer.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : tensor, array or None
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic RNN in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__.
    fw_initial_state : None or forward RNN State
        If None, `initial_state` is zero state.
    bw_initial_state : None or backward RNN State
        If None, `initial_state` is zero state.
    dropout : float
        The input and output keep probability (input_keep_prob, output_keep_prob).
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking Dense after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, 2 * n_hidden], for stacking multiple RNN after it.
    dynamic_rnn_init_args : dictionary
        The arguments for ``tf.nn.bidirectional_dynamic_rnn``.
    name : str
        A unique layer name.

    Attributes
    -----------------------
    outputs : tensor
        The output of this layer. (?, 2 * n_hidden)

    fw(bw)_final_state : tensor or StateTuple
        The finial state of this layer.
            - When `state_is_tuple` is `False`, it is the final hidden and cell states, `states.get_shape() = [?, 2 * n_hidden]`.
            - When `state_is_tuple` is `True`, it stores two elements: `(c, h)`.
            - In practice, you can get the final state after each iteration during training, then feed it to the initial state of next iteration.

    fw(bw)_initial_state : tensor or StateTuple
        The initial state of this layer.
            - In practice, you can set your state at the begining of each epoch or iteration according to your training procedure.

    batch_size : int or tensor
        It is an integer, if it is able to compute the `batch_size`; otherwise, tensor for dynamic batch size.

    sequence_length : a tensor or array
        The sequence lengths computed by Advanced Opt or the given sequence lengths, [batch_size].

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`Reshape`.

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__
    - `bidirectional_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/bidirectional_rnn.ipynb>`__

    """

    def __init__(
        self,
        cell_fn=None,
        cell_init_args=None,
        n_hidden=256,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length=None,
        fw_initial_state=None,
        bw_initial_state=None,
        dropout=None,
        n_layer=1,
        return_last=False,
        return_seq_2d=False,
        dynamic_rnn_init_args=None,
        name='bi_dyrnn_layer',
    ):
        if cell_fn is None:
            cell_fn = tf.contrib.rnn.BasicLSTMCell

        if cell_init_args and ('LSTM' in cell_fn.__name__):
            cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        self.cell_fn = cell_fn
        self.cell_init_args = cell_init_args
        self.n_hidden = n_hidden
        self.initializer = initializer
        self.sequence_length = sequence_length
        self.fw_initial_state = fw_initial_state
        self.bw_initial_state = bw_initial_state
        self.dropout = dropout
        self.n_layer = n_layer
        self.return_last = return_last
        self.return_seq_2d = return_seq_2d
        self.dynamic_rnn_init_args = dynamic_rnn_init_args
        self.name = name

        super(BiDynamicRNN, self).__init__(
            cell_init_args=cell_init_args,
            dynamic_rnn_init_args=dynamic_rnn_init_args,
        )

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_hidden: %d" % self.n_hidden)
        except AttributeError:
            pass

        try:
            additional_str.append("n_steps: %d" % self.n_steps)
        except AttributeError:
            pass

        try:
            additional_str.append("cell_fn: %s" % self.cell_fn.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("return_last: %s" % str(self.return_last))
        except AttributeError:
            pass

        try:
            additional_str.append("return_seq_2d: %s" % str(self.return_seq_2d))
        except AttributeError:
            pass

        try:
            additional_str.append("input shape: %s" % self._temp_data['inputs'].shape)
        except AttributeError:
            pass

        try:
            _dropout = str(self.dropout) if self._temp_data['is_train'] and self.dropout is not None else "disabled"
            additional_str.append("dropout: %s" % _dropout)
        except AttributeError:
            pass

        return self._str(additional_str)
        # logging.info(
        #     "BiDynamicRNN %s: n_hidden: %d in_dim: %d in_shape: %s cell_fn: %s dropout: %s n_layer: %d" % (
        #         self.name, n_hidden, self._temp_data['inputs'].get_shape().ndims, self._temp_data['inputs'].get_shape(),
        #         cell_fn.__name__, dropout, n_layer
        #     )
        # )

    def build(self):

        self._temp_data['dropout'] = self.dropout if self._temp_data['is_train'] else None

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self._temp_data['inputs'].get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self._temp_data['inputs'].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self._temp_data['inputs'])[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        self.batch_size = batch_size

        with tf.variable_scope(self.name, initializer=self.initializer) as vs:
            # Creats the cell function
            # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **self.cell_init_args) # HanSheng
            rnn_creator = lambda: self.cell_fn(num_units=self.n_hidden, **self.cell_init_args)

            # Apply dropout
            if self._temp_data['dropout'] is not None:
                if isinstance(self._temp_data['dropout'], (tuple, list)):
                    in_keep_prob = self._temp_data['dropout'][0]
                    out_keep_prob = self._temp_data['dropout'][1]
                elif isinstance(self._temp_data['dropout'], float):
                    in_keep_prob, out_keep_prob = self._temp_data['dropout'], self._temp_data['dropout']
                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of " "float)")
                try:
                    DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper
                except Exception:
                    DropoutWrapper_fn = tf.nn.rnn_cell.DropoutWrapper

                    # cell_instance_fn1=cell_instance_fn            # HanSheng
                    # cell_instance_fn=lambda: DropoutWrapper_fn(
                    #                     cell_instance_fn1(),
                    #                     input_keep_prob=in_keep_prob,
                    #                     output_keep_prob=out_keep_prob)
                cell_creator = lambda is_last=True: DropoutWrapper_fn(
                    rnn_creator(), input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob if is_last else 1.0
                )
            else:
                cell_creator = rnn_creator

            # if dropout:
            #     self.fw_cell = DropoutWrapper_fn(self.fw_cell, input_keep_prob=1.0, output_keep_prob=out_keep_prob)
            #     self.bw_cell = DropoutWrapper_fn(self.bw_cell, input_keep_prob=1.0, output_keep_prob=out_keep_prob)

            # self.fw_cell=cell_instance_fn()
            # self.bw_cell=cell_instance_fn()
            # Initial state of RNN

            # self.fw_initial_state = fw_initial_state
            # self.bw_initial_state = bw_initial_state
            # Computes sequence_length
            if self.sequence_length is None:

                self.sequence_length = retrieve_seq_length_op(
                    self._temp_data['inputs'] if isinstance(self._temp_data['inputs'], tf.Tensor) else tf.
                    stack(self._temp_data['inputs'])
                )

            if self.n_layer > 1:
                if self._temp_data['dropout'] is not None:
                    self.fw_cell = [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)]
                    self.bw_cell = [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)]

                else:
                    self.fw_cell = [cell_creator() for _ in range(self.n_layer)]
                    self.bw_cell = [cell_creator() for _ in range(self.n_layer)]

                outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
                    cells_fw=self.fw_cell,
                    cells_bw=self.bw_cell,
                    inputs=self._temp_data['inputs'],
                    sequence_length=self.sequence_length,
                    initial_states_fw=self.fw_initial_state,
                    initial_states_bw=self.bw_initial_state,
                    dtype=self._temp_data['inputs'].dtype,
                    **self.dynamic_rnn_init_args
                )

            else:
                self.fw_cell = cell_creator()
                self.bw_cell = cell_creator()
                outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_cell,
                    cell_bw=self.bw_cell,
                    inputs=self._temp_data['inputs'],
                    sequence_length=self.sequence_length,
                    initial_state_fw=self.fw_initial_state,
                    initial_state_bw=self.bw_initial_state,
                    dtype=self._temp_data['inputs'].dtype,
                    **self.dynamic_rnn_init_args
                )

            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            # logging.info("     n_params : %d" % (len(rnn_variables)))

            # Manage the outputs
            outputs = tf.concat(outputs, 2)

            if self.return_last:
                # [batch_size, 2 * n_hidden]
                raise NotImplementedError("Return last is not implemented yet.")
                # self._temp_data['outputs'] = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), 2 * n_hidden]
                if self.return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, 2 * n_hidden]
                    self._temp_data['outputs'] = tf.reshape(tf.concat(outputs, 1), [-1, 2 * self.n_hidden])

                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), 2 * n_hidden]
                    max_length = tf.shape(outputs)[1]
                    # batch_size = tf.shape(outputs)[0]

                    self._temp_data['outputs'] = tf.reshape(
                        tf.concat(outputs, 1),
                        [batch_size, max_length, 2 * self.n_hidden]
                    )

            self._temp_data['local_weights'] = rnn_variables

        # Initial states
        self._temp_data['fw_initial_state'] = self.fw_initial_state
        self._temp_data['bw_initial_state'] = self.bw_initial_state
        # Final states
        self._temp_data['fw_final_states'] = states_fw
        self._temp_data['bw_final_states'] = states_bw

        self._temp_data['sequence_length'] = self.sequence_length
