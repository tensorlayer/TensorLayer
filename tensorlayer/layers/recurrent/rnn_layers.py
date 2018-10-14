#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_inspect import getfullargspec

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'RNN',
    'BiRNN',
]


class RNN(Layer):
    """
    The :class:`RNN` class is a fixed length recurrent layer for implementing vanilla RNN,
    LSTM, GRU and etc.

    Parameters
    ----------
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell
            - See `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__
            - Note TF1.0+ and TF1.0- are different
    cell_init_args : dictionary
        The arguments for the cell function.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the model parameters.
    n_steps : int
        The fixed sequence length.
    initial_state : None or RNN State
        If None, `initial_state` is zero state.
    return_last : boolean
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking Dense after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : Tensor
        The output of this layer.

    final_state : Tensor or StateTuple
        The finial state of this layer.
            - When `state_is_tuple` is `False`, it is the final hidden and cell states, `states.get_shape() = [?, 2 * n_hidden]`.
            - When `state_is_tuple` is `True`, it stores two elements: `(c, h)`.
            - In practice, you can get the final state after each iteration during training, then feed it to the initial state of next iteration.

    initial_state : Tensor or StateTuple
        The initial state of this layer.
            - In practice, you can set your state at the begining of each epoch or iteration according to your training procedure.

    batch_size : int or Tensor
        It is an integer, if it is able to compute the `batch_size`; otherwise, tensor for dynamic batch size.

    Examples
    --------
    - For synced sequence input and output, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__

    - For encoding see below.

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 32
    >>> num_steps = 5
    >>> vocab_size = 3000
    >>> hidden_size = 256
    >>> keep_prob = 0.8
    >>> input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> net = tl.layers.EmbeddingInput(vocabulary_size=vocab_size,
    ...     embedding_size=hidden_size, name='embed')(input_data)
    >>> net = tl.layers.Dropout(keep=keep_prob, is_fix=True, name='drop1')(net)
    >>> net = tl.layers.RNN(cell_fn=tf.contrib.rnn.BasicLSTMCell,
    ...     n_hidden=hidden_size, n_steps=num_steps, return_last=False, name='lstm1')(net)
    >>> net = tl.layers.Dropout(keep=keep_prob, is_fix=True, name='drop2')(net)
    >>> net = tl.layers.RNN(cell_fn=tf.contrib.rnn.BasicLSTMCell,
    ...     n_hidden=hidden_size, n_steps=num_steps, return_last=True, name='lstm2')(net)
    >>> net = tl.layers.Dropout(keep=keep_prob, is_fix=True, name='drop3')(net)
    >>> net = tl.layers.Dense(n_units=vocab_size, name='output')(net)

    - For CNN+LSTM

    >>> image_size = 100
    >>> batch_size = 10
    >>> num_steps = 5
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
    >>> net = tl.layers.Input(name='in')(x)
    >>> net = tl.layers.Conv2d(32, (5, 5), (2, 2), tf.nn.relu, name='conv2d_1')(net)
    >>> net = tl.layers.MaxPool2d((2, 2), (2, 2), name='pool1')(net)
    >>> net = tl.layers.Conv2d(10, (5, 5), (2, 2), tf.nn.relu, name='conv2d_2')(net)
    >>> net = tl.layers.MaxPool2d((2, 2), (2, 2), name='pool2')(net)
    >>> net = tl.layers.Flatten(name='flatten')(net)
    >>> net = tl.layers.Reshape(shape=[-1, num_steps, int(net.outputs._shape[-1])])(net)
    >>> rnn = tl.layers.RNN(cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=200, n_steps=num_steps, return_last=False, return_seq_2d=True, name='rnn')(net)
    >>> net = tl.layers.Dense(rnn, 3, name='out')(net)

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`Reshape`.

    References
    ----------
    - `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`__
    - `tensorflow/python/ops/rnn.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py>`__
    - `tensorflow/python/ops/rnn_cell.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py>`__
    - see TensorFlow tutorial ``ptb_word_lm.py``, TensorLayer tutorials ``tutorial_ptb_lstm*.py`` and ``tutorial_generate_text.py``

    """

    def __init__(
        self,
        cell_fn=None,
        cell_init_args=None,
        n_hidden=100,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        n_steps=5,
        initial_state=None,
        return_last=False,
        return_seq_2d=False,
        name='rnn',
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
        self.n_steps = n_steps
        self.initial_state = initial_state
        self.return_last = return_last
        self.return_seq_2d = return_seq_2d
        self.name = name

        super(RNN, self).__init__(cell_init_args=cell_init_args)

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
            additional_str.append("num weights: %d" % self.n_weights)
        except AttributeError:
            pass

        return self._str(additional_str)

        # logging.info(
        #     "RNN %s: n_hidden: %d n_steps: %d in_dim: %d in_shape: %s cell_fn: %s " % (
        #         self.name, n_hidden, n_steps, self._temp_data['inputs'].get_shape().ndims,
        #         self._temp_data['inputs'].get_shape(), cell_fn.__name__
        #     )
        # )

    def build(self):

        if 'GRU' in self.cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning('pop state_is_tuple fails.')

        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self._temp_data['inputs'].get_shape().with_rank(2)
        # self._temp_data['inputs'].get_shape().with_rank(3)

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self._temp_data['inputs'].get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        # is_reshape : boolean (deprecate)
        #     Reshape the inputs to 3 dimension tensor.\n
        #     If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        #     If input is [batch_size * n_steps, n_features], we need to reshape it.
        # if is_reshape:
        #     self._temp_data['inputs'] = tf.reshape(self._temp_data['inputs'], shape=[-1, n_steps, int(self._temp_data['inputs']._shape[-1])])

        fixed_batch_size = self._temp_data['inputs'].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       RNN batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self._temp_data['inputs'])[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        self.batch_size = batch_size

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []

        if 'reuse' in getfullargspec(self.cell_fn.__init__).args:
            self.cell = cell = self.cell_fn(
                num_units=self.n_hidden, reuse=tf.get_variable_scope().reuse, **self.cell_init_args
            )
        else:
            self.cell = cell = self.cell_fn(num_units=self.n_hidden, **self.cell_init_args)

        if self.initial_state is None:
            self.initial_state = cell.zero_state(
                batch_size, dtype=self._temp_data['inputs'].dtype
            )  #dtype=tf.float32)  # 1.2.3

        state = self.initial_state

        with tf.variable_scope(self.name, initializer=self.initializer) as vs:
            for time_step in range(self.n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self._temp_data['inputs'][:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            # rnn_variables = get_collection_trainable(self.name)
            self._temp_data['local_weights'] = rnn_variables
            # logging.info("     n_params : %d" % (len(rnn_variables)))
            self.n_weights = len(rnn_variables)

            if self.return_last:
                # 2D Tensor [batch_size, n_hidden]
                self._temp_data['outputs'] = outputs[-1]
            else:
                if self.return_seq_2d:
                    # PTB tutorial: stack dense layer after that, or compute the cost from the output
                    # 2D Tensor [n_example, n_hidden]

                    self._temp_data['outputs'] = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden])

                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]

                    self._temp_data['outputs'] = tf.reshape(tf.concat(outputs, 1), [-1, self.n_steps, self.n_hidden])

        self._temp_data['final_state'] = state
        self._temp_data['initial_state'] = self.initial_state


class BiRNN(Layer):
    """
    The :class:`BiRNN` class is a fixed length Bidirectional recurrent layer.

    Parameters
    ----------
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell.
            - See `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__.
            - Note TF1.0+ and TF1.0- are different.
    cell_init_args : dictionary or None
        The arguments for the cell function.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the model parameters.
    n_steps : int
        The fixed sequence length.
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
            - If True, return 2D Tensor [n_example, n_hidden], for stacking Dense after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : tensor
        The output of this layer.
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

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features]. If not, please see :class:`Reshape`.
    For predicting, the sequence length has to be the same with the sequence length of training, while, for normal
    RNN, we can use sequence length of 1 for predicting.

    References
    ----------
    `Source <https://github.com/akaraspt/deepsleep/blob/master/deepsleep/model.py>`__

    """

    def __init__(
        self,
        cell_fn=None,
        cell_init_args=None,
        n_hidden=100,
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        n_steps=5,
        fw_initial_state=None,
        bw_initial_state=None,
        dropout=None,
        n_layer=1,
        return_last=False,
        return_seq_2d=False,
        name='birnn',
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
        self.n_steps = n_steps
        self.fw_initial_state = fw_initial_state
        self.bw_initial_state = bw_initial_state
        self.dropout = dropout
        self.n_layer = n_layer
        self.return_last = return_last
        self.return_seq_2d = return_seq_2d
        self.name = name

        super(BiRNN, self).__init__(cell_init_args=cell_init_args)

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
            additional_str.append("num weights: %d" % self.n_weights)
        except AttributeError:
            pass

        try:
            _dropout = str(self.dropout) if self._temp_data['is_train'] and self.dropout is not None else "disabled"
            additional_str.append("dropout: %s" % _dropout)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['dropout'] = self.dropout if self._temp_data['is_train'] else None

        fixed_batch_size = self._temp_data['inputs'].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            self.batch_size = fixed_batch_size.value
            logging.info("       RNN batch_size (concurrent processes): %d" % self.batch_size)

        else:
            self.batch_size = array_ops.shape(self._temp_data['inputs'])[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self._temp_data['inputs'].get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.variable_scope(self.name, initializer=self.initializer) as vs:
            rnn_creator = lambda: self.cell_fn(num_units=self.n_hidden, **self.cell_init_args)
            # Apply dropout
            if self._temp_data['dropout'] is not None:

                if isinstance(self._temp_data['dropout'], (tuple, list)):  # type(dropout) in [tuple, list]:
                    in_keep_prob = self._temp_data['dropout'][0]
                    out_keep_prob = self._temp_data['dropout'][1]

                elif isinstance(self._temp_data['dropout'], float):
                    in_keep_prob, out_keep_prob = self._temp_data['dropout'], self._temp_data['dropout']

                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of " "float)")

                DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper

                cell_creator = lambda is_last=True: DropoutWrapper_fn(
                    rnn_creator(), input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob if is_last else 1.0
                )

            else:
                # logging.info('disable dropout as is_train is False')
                cell_creator = rnn_creator

            self.fw_cell = cell_creator()
            self.bw_cell = cell_creator()

            # Apply multiple layers
            if self.n_layer > 1:
                MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell

                if self._temp_data['dropout'] is not None:
                    try:
                        self.fw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)],
                            state_is_tuple=True
                        )
                        self.bw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)],
                            state_is_tuple=True
                        )
                    except Exception:
                        self.fw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)]
                        )
                        self.bw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == self.n_layer - 1) for i in range(self.n_layer)]
                        )
                else:
                    try:
                        self.fw_cell = MultiRNNCell_fn(
                            [cell_creator() for _ in range(self.n_layer)], state_is_tuple=True
                        )
                        self.bw_cell = MultiRNNCell_fn(
                            [cell_creator() for _ in range(self.n_layer)], state_is_tuple=True
                        )
                    except Exception:
                        self.fw_cell = MultiRNNCell_fn([cell_creator() for _ in range(self.n_layer)])
                        self.bw_cell = MultiRNNCell_fn([cell_creator() for _ in range(self.n_layer)])

            # Initial state of RNN
            if self.fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(
                    self.batch_size, dtype=self._temp_data['inputs'].dtype
                )  # dtype=tf.float32)
            # else:
            #     self.fw_initial_state = self.fw_initial_state
            if self.bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(
                    self.batch_size, dtype=self._temp_data['inputs'].dtype
                )  # dtype=tf.float32)
            # else:
            #     self.bw_initial_state = bw_initial_state
            # exit()
            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(self._temp_data['inputs'], axis=1)

            bidirectional_rnn_fn = tf.contrib.rnn.static_bidirectional_rnn

            outputs, fw_state, bw_state = bidirectional_rnn_fn(  # outputs, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if self.return_last:
                raise Exception("Do not support return_last at the moment.")
                # self._temp_data['outputs'] = outputs[-1]

            else:
                self._temp_data['outputs'] = outputs
                if self.return_seq_2d:
                    # 2D Tensor [n_example, n_hidden]
                    self._temp_data['outputs'] = tf.reshape(tf.concat(outputs, 1), [-1, self.n_hidden * 2])

                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]

                    self._temp_data['outputs'] = tf.reshape(
                        tf.concat(outputs, 1), [-1, self.n_steps, self.n_hidden * 2]
                    )
            # Retrieve just the RNN variables.
            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # Initial states
        self._temp_data['fw_initial_state'] = self.fw_initial_state
        self._temp_data['bw_initial_state'] = self.bw_initial_state
        # Final states
        self._temp_data['fw_final_state'] = fw_state
        self._temp_data['bw_final_state'] = bw_state
