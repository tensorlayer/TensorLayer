#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_inspect import getfullargspec
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'RNNLayer',
    'BiRNNLayer',
    'ConvRNNCell',
    'BasicConvLSTMCell',
    'ConvLSTMLayer',
    'advanced_indexing_op',
    'retrieve_seq_length_op',
    'retrieve_seq_length_op2',
    'retrieve_seq_length_op3',
    'target_mask_op',
    'DynamicRNNLayer',
    'BiDynamicRNNLayer',
    'Seq2Seq',
]


class RNNLayer(Layer):
    """
    The :class:`RNNLayer` class is a fixed length recurrent layer for implementing vanilla RNN,
    LSTM, GRU and etc.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
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
    >>> is_train = True
    >>> input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> net = tl.layers.EmbeddingInputlayer(inputs=input_data, vocabulary_size=vocab_size,
    ...     embedding_size=hidden_size, name='embed')
    >>> net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop1')
    >>> net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell,
    ...     n_hidden=hidden_size, n_steps=num_steps, return_last=False, name='lstm1')
    >>> net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop2')
    >>> net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell,
    ...     n_hidden=hidden_size, n_steps=num_steps, return_last=True, name='lstm2')
    >>> net = tl.layers.DropoutLayer(net, keep=keep_prob, is_fix=True, is_train=is_train, name='drop3')
    >>> net = tl.layers.DenseLayer(net, n_units=vocab_size, name='output')

    - For CNN+LSTM

    >>> image_size = 100
    >>> batch_size = 10
    >>> num_steps = 5
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
    >>> net = tl.layers.InputLayer(x, name='in')
    >>> net = tl.layers.Conv2d(net, 32, (5, 5), (2, 2), tf.nn.relu, name='cnn1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), name='pool1')
    >>> net = tl.layers.Conv2d(net, 10, (5, 5), (2, 2), tf.nn.relu, name='cnn2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), name='pool2')
    >>> net = tl.layers.FlattenLayer(net, name='flatten')
    >>> net = tl.layers.ReshapeLayer(net, shape=[-1, num_steps, int(net.outputs._shape[-1])])
    >>> rnn = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.BasicLSTMCell, n_hidden=200, n_steps=num_steps, return_last=False, return_seq_2d=True, name='rnn')
    >>> net = tl.layers.DenseLayer(rnn, 3, name='out')

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`ReshapeLayer`.

    References
    ----------
    - `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`__
    - `tensorflow/python/ops/rnn.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py>`__
    - `tensorflow/python/ops/rnn_cell.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py>`__
    - see TensorFlow tutorial ``ptb_word_lm.py``, TensorLayer tutorials ``tutorial_ptb_lstm*.py`` and ``tutorial_generate_text.py``

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            cell_fn,
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
            raise Exception("Please put in cell_fn")

        super(RNNLayer, self).__init__(prev_layer=prev_layer, cell_init_args=cell_init_args, name=name)

        if 'GRU' in cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning('pop state_is_tuple fails.')

        logging.info(
            "RNNLayer %s: n_hidden: %d n_steps: %d in_dim: %d in_shape: %s cell_fn: %s " %
            (self.name, n_hidden, n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__)
        )

        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        # is_reshape : boolean (deprecate)
        #     Reshape the inputs to 3 dimension tensor.\n
        #     If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        #     If input is [batch_size * n_steps, n_features], we need to reshape it.
        # if is_reshape:
        #     self.inputs = tf.reshape(self.inputs, shape=[-1, n_steps, int(self.inputs._shape[-1])])

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       RNN batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self.inputs)[0]
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

        if 'reuse' in getfullargspec(cell_fn.__init__).args:
            self.cell = cell = cell_fn(num_units=n_hidden, reuse=tf.get_variable_scope().reuse, **self.cell_init_args)
        else:
            self.cell = cell = cell_fn(num_units=n_hidden, **self.cell_init_args)

        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=LayersConfig.tf_dtype)  #dtype=tf.float32)  # 1.2.3

        state = self.initial_state

        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            logging.info("     n_params : %d" % (len(rnn_variables)))

            if return_last:
                # 2D Tensor [batch_size, n_hidden]
                self.outputs = outputs[-1]
            else:
                if return_seq_2d:
                    # PTB tutorial: stack dense layer after that, or compute the cost from the output
                    # 2D Tensor [n_example, n_hidden]

                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])

                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]

                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_steps, n_hidden])

        self.final_state = state

        self._add_layers(self.outputs)
        self._add_params(rnn_variables)


class BiRNNLayer(Layer):
    """
    The :class:`BiRNNLayer` class is a fixed length Bidirectional recurrent layer.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
    dropout : tuple of float or int
        The input and output keep probability (input_keep_prob, output_keep_prob).
        If one int, input and output keep probability are the same.
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
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
    Input dimension should be rank 3 : [batch_size, n_steps, n_features]. If not, please see :class:`ReshapeLayer`.
    For predicting, the sequence length has to be the same with the sequence length of training, while, for normal
    RNN, we can use sequence length of 1 for predicting.

    References
    ----------
    `Source <https://github.com/akaraspt/deepsleep/blob/master/deepsleep/model.py>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            cell_fn,
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
        super(BiRNNLayer, self).__init__(prev_layer=prev_layer, cell_init_args=cell_init_args, name=name)

        if self.cell_init_args:
            self.cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if 'GRU' in cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        if cell_fn is None:
            raise Exception("Please put in cell_fn")

        logging.info(
            "BiRNNLayer %s: n_hidden: %d n_steps: %d in_dim: %d in_shape: %s cell_fn: %s dropout: %s n_layer: %d " % (
                self.name, n_hidden, n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__,
                dropout, n_layer
            )
        )

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            self.batch_size = fixed_batch_size.value
            logging.info("       RNN batch_size (concurrent processes): %d" % self.batch_size)

        else:
            self.batch_size = array_ops.shape(self.inputs)[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.variable_scope(name, initializer=initializer) as vs:
            rnn_creator = lambda: cell_fn(num_units=n_hidden, **self.cell_init_args)
            # Apply dropout
            if dropout:

                if isinstance(dropout, (tuple, list)):  # type(dropout) in [tuple, list]:
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]

                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout

                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of " "float)")

                DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper

                cell_creator = lambda is_last=True: DropoutWrapper_fn(
                    rnn_creator(), input_keep_prob=in_keep_prob, output_keep_prob=out_keep_prob if is_last else 1.0
                )

            else:
                cell_creator = rnn_creator

            self.fw_cell = cell_creator()
            self.bw_cell = cell_creator()

            # Apply multiple layers
            if n_layer > 1:
                MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell

                if dropout:
                    try:
                        self.fw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)], state_is_tuple=True
                        )
                        self.bw_cell = MultiRNNCell_fn(
                            [cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)], state_is_tuple=True
                        )
                    except Exception:
                        self.fw_cell = MultiRNNCell_fn([cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)])
                        self.bw_cell = MultiRNNCell_fn([cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)])
                else:
                    try:
                        self.fw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)], state_is_tuple=True)
                        self.bw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)], state_is_tuple=True)
                    except Exception:
                        self.fw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)])
                        self.bw_cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)])

            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(
                    self.batch_size, dtype=LayersConfig.tf_dtype
                )  # dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(
                    self.batch_size, dtype=LayersConfig.tf_dtype
                )  # dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # exit()
            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unstack(self.inputs, axis=1)

            bidirectional_rnn_fn = tf.contrib.rnn.static_bidirectional_rnn

            outputs, fw_state, bw_state = bidirectional_rnn_fn(  # outputs, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if return_last:
                raise Exception("Do not support return_last at the moment.")
                # self.outputs = outputs[-1]
            else:
                self.outputs = outputs
                if return_seq_2d:
                    # 2D Tensor [n_example, n_hidden]
                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden * 2])

                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]

                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_steps, n_hidden * 2])

            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

            # Retrieve just the RNN variables.
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        logging.info("     n_params : %d" % (len(rnn_variables)))

        self._add_layers(self.outputs)
        self._add_params(rnn_variables)


class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN Cell."""

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state."""
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype=LayersConfig.tf_dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros

        """
        shape = self.shape
        num_features = self.num_features
        # TODO : TypeError: 'NoneType' object is not subscriptable
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2], dtype=dtype)
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell.

    Parameters
    -----------
    shape : tuple of int
        The height and width of the cell.
    filter_size : tuple of int
        The height and width of the filter
    num_features : int
        The hidden size of the cell
    forget_bias : float
        The bias added to forget gates (see above).
    input_size : int
        Deprecated and unused.
    state_is_tuple : boolen
        If True, accepted and returned states are 2-tuples of the `c_state` and `m_state`.
        If False, they are concatenated along the column axis. The latter behavior will soon be deprecated.
    act : activation function
        The activation function of this layer, tanh as default.

    """

    def __init__(
            self, shape, filter_size, num_features, forget_bias=1.0, input_size=None, state_is_tuple=False,
            act=tf.nn.tanh
    ):
        """Initialize the basic Conv LSTM cell."""
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = act

    @property
    def state_size(self):
        """State size of the LSTMStateTuple."""
        return (LSTMStateTuple(self._num_units, self._num_units) if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        """Number of units in outputs."""
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                # print state
                # c, h = tf.split(3, 2, state)
                c, h = tf.split(state, 2, 3)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            # i, j, f, o = tf.split(3, 4, concat)
            i, j, f, o = tf.split(concat, 4, 3)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:

    Parameters
    ----------
    args : tensor
        4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size : tuple of int
        Filter height and width.
    num_features : int
        Nnumber of features.
    bias_start : float
        Starting value to initialize the bias; 0 by default.
    scope : VariableScope
        For the created subgraph; defaults to "Linear".

    Returns
    --------
    - A 4D Tensor with shape [batch h w num_features]

    Raises
    -------
    - ValueError : if some of the arguments has unspecified or wrong shape.

    """
    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype
        )
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(args, 3), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
    return res + bias_term


class ConvLSTMLayer(Layer):
    """A fixed length Convolutional LSTM layer.

    See this `paper <https://arxiv.org/abs/1506.04214>`__ .

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    cell_shape : tuple of int
        The shape of each cell width * height
    filter_size : tuple of int
        The size of filter width * height
    cell_fn : a convolutional RNN cell
        Cell function like :class:`BasicConvLSTMCell`
    feature_map : int
        The number of feature map in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : int
        The sequence length.
    initial_state : None or ConvLSTM State
        If None, `initial_state` is zero state.
    return_last : boolean
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output".
            - If False, return all outputs, "Synced sequence input and output".
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : tensor
        The output of this RNN. return_last = False, outputs = all cell_output, which is the hidden state.
        cell_output.get_shape() = (?, h, w, c])

    final_state : tensor or StateTuple
        The finial state of this layer.
            - When state_is_tuple = False, it is the final hidden and cell states,
            - When state_is_tuple = True, You can get the final state after each iteration during training, then feed it to the initial state of next iteration.

    initial_state : tensor or StateTuple
        It is the initial state of this ConvLSTM layer, you can use it to initialize
        your state at the beginning of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            cell_shape=None,
            feature_map=1,
            filter_size=(3, 3),
            cell_fn=BasicConvLSTMCell,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            n_steps=5,
            initial_state=None,
            return_last=False,
            return_seq_2d=False,
            name='convlstm',
    ):
        super(ConvLSTMLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "ConvLSTMLayer %s: feature_map: %d, n_steps: %d, "
            "in_dim: %d %s, cell_fn: %s " %
            (self.name, feature_map, n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__)
        )
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 5 [batch_size, n_steps(max), h, w, c]
        try:
            self.inputs.get_shape().with_rank(5)
        except Exception:
            raise Exception(
                "RNN : Input dimension should be rank 5 : [batch_size, n_steps, input_x, "
                "input_y, feature_map]"
            )

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("     RNN batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self.inputs)[0]
            logging.info("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size
        outputs = []
        self.cell = cell = cell_fn(shape=cell_shape, filter_size=filter_size, num_features=feature_map)

        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=LayersConfig.tf_dtype)
        else:
            self.initial_state = initial_state

        state = self.initial_state

        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :, :, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

            logging.info(" n_params : %d" % (len(rnn_variables)))

            if return_last:
                # 2D Tensor [batch_size, n_hidden]
                self.outputs = outputs[-1]
            else:
                if return_seq_2d:
                    # PTB tutorial: stack dense layer after that, or compute the cost from the output
                    # 4D Tensor [n_example, h, w, c]
                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, cell_shape[0] * cell_shape[1] * feature_map])
                else:
                    # <akara>: stack more RNN layer after that
                    # 5D Tensor [n_example/n_steps, n_steps, h, w, c]
                    self.outputs = tf.reshape(
                        tf.concat(outputs, 1), [-1, n_steps, cell_shape[0], cell_shape[1], feature_map]
                    )

        self.final_state = state

        self._add_layers(self.outputs)
        self._add_params(rnn_variables)


# Advanced Ops for Dynamic RNN
def advanced_indexing_op(inputs, index):
    """Advanced Indexing for Sequences, returns the outputs by given sequence lengths.
    When return the last output :class:`DynamicRNNLayer` uses it to get the last outputs with the sequence lengths.

    Parameters
    -----------
    inputs : tensor for data
        With shape of [batch_size, n_step(max), n_features]
    index : tensor for indexing
        Sequence length in Dynamic RNN. [batch_size]

    Examples
    ---------
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size, max_length, n_features = 3, 5, 2
    >>> z = np.random.uniform(low=-1, high=1, size=[batch_size, max_length, n_features]).astype(np.float32)
    >>> b_z = tf.constant(z)
    >>> sl = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    >>> o = advanced_indexing_op(b_z, sl)
    >>>
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>>
    >>> order = np.asarray([1,1,2])
    >>> print("real",z[0][order[0]-1], z[1][order[1]-1], z[2][order[2]-1])
    >>> y = sess.run([o], feed_dict={sl:order})
    >>> print("given",order)
    >>> print("out", y)
    real [-0.93021595  0.53820813] [-0.92548317 -0.77135968] [ 0.89952248  0.19149846]
    given [1 1 2]
    out [array([[-0.93021595,  0.53820813],
                [-0.92548317, -0.77135968],
                [ 0.89952248,  0.19149846]], dtype=float32)]

    References
    -----------
    - Modified from TFlearn (the original code is used for fixed length rnn), `references <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__.

    """
    batch_size = tf.shape(inputs)[0]
    # max_length = int(inputs.get_shape()[1])    # for fixed length rnn, length is given
    max_length = tf.shape(inputs)[1]  # for dynamic_rnn, length is unknown
    dim_size = int(inputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(inputs, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant


def retrieve_seq_length_op(data):
    """An op to compute the length of a sequence from input shape of [batch_size, n_step(max), n_features],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max), n_features] with zero padding on right hand side.

    Examples
    ---------
    >>> data = [[[1],[2],[0],[0],[0]],
    ...         [[1],[2],[3],[0],[0]],
    ...         [[1],[2],[6],[1],[0]]]
    >>> data = np.asarray(data)
    >>> print(data.shape)
    (3, 5, 1)
    >>> data = tf.constant(data)
    >>> sl = retrieve_seq_length_op(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> y = sl.eval()
    [2 3 4]

    Multiple features
    >>> data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
    ...         [[2,3],[2,4],[3,2],[0,0],[0,0]],
    ...         [[3,3],[2,2],[5,3],[1,2],[0,0]]]
    >>> print(sl)
    [4 3 4]

    References
    ------------
    Borrow from `TFlearn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__.

    """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), 2))
        length = tf.reduce_sum(used, 1)

        return tf.cast(length, tf.int32)


def retrieve_seq_length_op2(data):
    """An op to compute the length of a sequence, from input shape of [batch_size, n_step(max)],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] with zero padding on right hand side.

    Examples
    --------
    >>> data = [[1,2,0,0,0],
    ...         [1,2,3,0,0],
    ...         [1,2,6,1,0]]
    >>> o = retrieve_seq_length_op2(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> print(o.eval())
    [2 3 4]

    """
    return tf.reduce_sum(tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), 1)


def retrieve_seq_length_op3(data, pad_val=0):  # HangSheng: return tensor for sequence length, if input is tf.string
    """Return tensor for sequence length, if input is ``tf.string``.

    """
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.reduce_sum(tf.cast(tf.reduce_any(tf.not_equal(data, pad_val), axis=2), dtype=tf.int32), 1)
    elif data_shape_size == 2:
        return tf.reduce_sum(tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32), 1)
    elif data_shape_size == 1:
        raise ValueError("retrieve_seq_length_op3: data has wrong shape!")
    else:
        raise ValueError(
            "retrieve_seq_length_op3: handling data_shape_size %s hasn't been implemented!" % (data_shape_size)
        )


def target_mask_op(data, pad_val=0):  # HangSheng: return tensor for mask,if input is tf.string
    """Return tensor for mask, if input is ``tf.string``.

    """
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.cast(tf.reduce_any(tf.not_equal(data, pad_val), axis=2), dtype=tf.int32)
    elif data_shape_size == 2:
        return tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32)
    elif data_shape_size == 1:
        raise ValueError("target_mask_op: data has wrong shape!")
    else:
        raise ValueError("target_mask_op: handling data_shape_size %s hasn't been implemented!" % (data_shape_size))


class DynamicRNNLayer(Layer):
    """
    The :class:`DynamicRNNLayer` class is a dynamic recurrent layer, see ``tf.nn.dynamic_rnn``.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
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
    dropout : tuple of float or int
        The input and output keep probability (input_keep_prob, output_keep_prob).
            - If one int, input and output keep probability are the same.
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean or None
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
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
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    Examples
    --------
    Synced sequence input and output, for loss function see ``tl.cost.cross_entropy_seq_with_mask``.

    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
    >>> net = tl.layers.EmbeddingInputlayer(
    ...             inputs=input_seqs,
    ...             vocabulary_size=vocab_size,
    ...             embedding_size=embedding_size,
    ...             name='embedding')
    >>> net = tl.layers.DynamicRNNLayer(net,
    ...             cell_fn=tf.contrib.rnn.BasicLSTMCell, # for TF0.2 use tf.nn.rnn_cell.BasicLSTMCell,
    ...             n_hidden=embedding_size,
    ...             dropout=(0.7 if is_train else None),
    ...             sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
    ...             return_last=False,                    # for encoder, set to True
    ...             return_seq_2d=True,                   # stack denselayer or compute cost after it
    ...             name='dynamicrnn')
    >>> net = tl.layers.DenseLayer(net, n_units=vocab_size, name="output")

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__
    - `dynamic_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb>`__
    - `tf.nn.dynamic_rnn <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md>`__
    - `tflearn rnn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__
    - ``tutorial_dynamic_rnn.py``

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            cell_fn,  #tf.nn.rnn_cell.LSTMCell,
            cell_init_args=None,
            n_hidden=256,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            sequence_length=None,
            initial_state=None,
            dropout=None,
            n_layer=1,
            return_last=None,
            return_seq_2d=False,
            dynamic_rnn_init_args=None,
            name='dyrnn',
    ):
        if cell_fn is None:
            raise Exception("Please put in cell_fn")

        super(DynamicRNNLayer, self).__init__(
            prev_layer=prev_layer, cell_init_args=cell_init_args, dynamic_rnn_init_args=dynamic_rnn_init_args, name=name
        )

        if self.cell_init_args:
            self.cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True

        if 'GRU' in cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        if return_last is None:
            return_last = True

        logging.info(
            "DynamicRNNLayer %s: n_hidden: %d, in_dim: %d in_shape: %s cell_fn: %s dropout: %s n_layer: %d" % (
                self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout,
                n_layer
            )
        )

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self.inputs)[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        self.batch_size = batch_size

        # Creats the cell function
        # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **self.cell_init_args) # HanSheng
        rnn_creator = lambda: cell_fn(num_units=n_hidden, **self.cell_init_args)

        # Apply dropout
        if dropout:
            if isinstance(dropout, (tuple, list)):
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]

            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout

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
        if n_layer > 1:
            try:
                MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell
            except Exception:
                MultiRNNCell_fn = tf.nn.rnn_cell.MultiRNNCell

            # cell_instance_fn2=cell_instance_fn # HanSheng
            if dropout:
                try:
                    # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)], state_is_tuple=True) # HanSheng
                    self.cell = MultiRNNCell_fn(
                        [cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)], state_is_tuple=True
                    )
                except Exception:  # when GRU
                    # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)]) # HanSheng
                    self.cell = MultiRNNCell_fn([cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)])
            else:
                try:
                    self.cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)], state_is_tuple=True)
                except Exception:  # when GRU
                    self.cell = MultiRNNCell_fn([cell_creator() for _ in range(n_layer)])

        # self.cell=cell_instance_fn() # HanSheng

        # Initialize initial_state
        if initial_state is None:
            self.initial_state = self.cell.zero_state(batch_size, dtype=LayersConfig.tf_dtype)  # dtype=tf.float32)
        else:
            self.initial_state = initial_state

        # Computes sequence_length
        if sequence_length is None:

            sequence_length = retrieve_seq_length_op(
                self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs)
            )

        # Main - Computes outputs and last_states
        with tf.variable_scope(name, initializer=initializer) as vs:
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=self.cell,
                # inputs=X
                inputs=self.inputs,
                # dtype=tf.float64,
                sequence_length=sequence_length,
                initial_state=self.initial_state,
                **self.dynamic_rnn_init_args
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            # logging.info("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            if return_last:
                # [batch_size, n_hidden]
                # outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
                self.outputs = advanced_indexing_op(outputs, sequence_length)

            else:
                # [batch_size, n_step(max), n_hidden]
                # self.outputs = result[0]["outputs"]
                # self.outputs = outputs    # it is 3d, but it is a list
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, n_hidden]
                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])

                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), n_hidden]
                    max_length = tf.shape(outputs)[1]
                    batch_size = tf.shape(outputs)[0]

                    self.outputs = tf.reshape(tf.concat(outputs, 1), [batch_size, max_length, n_hidden])
                    # self.outputs = tf.reshape(tf.concat(1, outputs), [-1, max_length, n_hidden])

        # Final state
        self.final_state = last_states

        self.sequence_length = sequence_length

        self._add_layers(self.outputs)
        self._add_params(rnn_variables)


class BiDynamicRNNLayer(Layer):
    """
    The :class:`BiDynamicRNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
    dropout : tuple of float or int
        The input and output keep probability (input_keep_prob, output_keep_prob).
            - If one int, input and output keep probability are the same.
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking DenseLayer after it.
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
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__
    - `bidirectional_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/bidirectional_rnn.ipynb>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            cell_fn,  #tf.nn.rnn_cell.LSTMCell,
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
        super(BiDynamicRNNLayer, self).__init__(
            prev_layer=prev_layer, cell_init_args=cell_init_args, dynamic_rnn_init_args=dynamic_rnn_init_args, name=name
        )

        if self.cell_init_args:
            self.cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if 'GRU' in cell_fn.__name__:
            try:
                self.cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        if cell_fn is None:
            raise Exception("Please put in cell_fn")

        logging.info(
            "BiDynamicRNNLayer %s: n_hidden: %d in_dim: %d in_shape: %s cell_fn: %s dropout: %s n_layer: %d" % (
                self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout,
                n_layer
            )
        )

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       batch_size (concurrent processes): %d" % batch_size)

        else:
            batch_size = array_ops.shape(self.inputs)[0]
            logging.info("       non specified batch_size, uses a tensor instead.")

        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **self.cell_init_args) # HanSheng
            rnn_creator = lambda: cell_fn(num_units=n_hidden, **self.cell_init_args)

            # Apply dropout
            if dropout:
                if isinstance(dropout, (tuple, list)):
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]
                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout
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

            self.fw_initial_state = fw_initial_state
            self.bw_initial_state = bw_initial_state
            # Computes sequence_length
            if sequence_length is None:

                sequence_length = retrieve_seq_length_op(
                    self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs)
                )

            if n_layer > 1:
                if dropout:
                    self.fw_cell = [cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)]
                    self.bw_cell = [cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)]

                else:
                    self.fw_cell = [cell_creator() for _ in range(n_layer)]
                    self.bw_cell = [cell_creator() for _ in range(n_layer)]

                outputs, states_fw, states_bw = stack_bidirectional_dynamic_rnn(
                    cells_fw=self.fw_cell, cells_bw=self.bw_cell, inputs=self.inputs, sequence_length=sequence_length,
                    initial_states_fw=self.fw_initial_state, initial_states_bw=self.bw_initial_state,
                    dtype=LayersConfig.tf_dtype, **self.dynamic_rnn_init_args
                )

            else:
                self.fw_cell = cell_creator()
                self.bw_cell = cell_creator()
                outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_cell, cell_bw=self.bw_cell, inputs=self.inputs, sequence_length=sequence_length,
                    initial_state_fw=self.fw_initial_state, initial_state_bw=self.bw_initial_state,
                    dtype=LayersConfig.tf_dtype, **self.dynamic_rnn_init_args
                )

            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            logging.info("     n_params : %d" % (len(rnn_variables)))

            # Manage the outputs
            outputs = tf.concat(outputs, 2)

            if return_last:
                # [batch_size, 2 * n_hidden]
                raise NotImplementedError("Return last is not implemented yet.")
                # self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), 2 * n_hidden]
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, 2 * n_hidden]
                    self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, 2 * n_hidden])

                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), 2 * n_hidden]
                    max_length = tf.shape(outputs)[1]
                    batch_size = tf.shape(outputs)[0]

                    self.outputs = tf.reshape(tf.concat(outputs, 1), [batch_size, max_length, 2 * n_hidden])

        # Final state
        self.fw_final_states = states_fw
        self.bw_final_states = states_bw

        self.sequence_length = sequence_length

        self._add_layers(self.outputs)
        self._add_params(rnn_variables)


class Seq2Seq(Layer):
    """
    The :class:`Seq2Seq` class is a simple :class:`DynamicRNNLayer` based Seq2seq layer without using `tl.contrib.seq2seq <https://www.tensorflow.org/api_guides/python/contrib.seq2seq>`__.
    See `Model <https://camo.githubusercontent.com/9e88497fcdec5a9c716e0de5bc4b6d1793c6e23f/687474703a2f2f73757269796164656570616e2e6769746875622e696f2f696d672f736571327365712f73657132736571322e706e67>`__
    and `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`__.

    - Please check this example `Chatbot in 200 lines of code <https://github.com/tensorlayer/seq2seq-chatbot>`__.
    - The Author recommends users to read the source code of :class:`DynamicRNNLayer` and :class:`Seq2Seq`.

    Parameters
    ----------
    net_encode_in : :class:`Layer`
        Encode sequences, [batch_size, None, n_features].
    net_decode_in : :class:`Layer`
        Decode sequences, [batch_size, None, n_features].
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
    >>> net_out.print_params(False)

    """

    def __init__(
            self,
            net_encode_in,
            net_decode_in,
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
        super(Seq2Seq, self).__init__(prev_layer=None, cell_init_args=cell_init_args, name=name)

        if self.cell_init_args:
            self.cell_init_args['state_is_tuple'] = True  # 'use_peepholes': True,

        if cell_fn is None:
            raise ValueError("cell_fn cannot be set to None")

        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")

        logging.info(
            "[*] Seq2Seq %s: n_hidden: %d cell_fn: %s dropout: %s n_layer: %d" %
            (self.name, n_hidden, cell_fn.__name__, dropout, n_layer)
        )

        with tf.variable_scope(name):
            # tl.layers.set_name_reuse(reuse)
            # network = InputLayer(self.inputs, name=name+'/input')
            network_encode = DynamicRNNLayer(
                net_encode_in, cell_fn=cell_fn, cell_init_args=self.cell_init_args, n_hidden=n_hidden,
                initializer=initializer, initial_state=initial_state_encode, dropout=dropout, n_layer=n_layer,
                sequence_length=encode_sequence_length, return_last=False, return_seq_2d=True, name='encode'
            )
            # vs.reuse_variables()
            # tl.layers.set_name_reuse(True)
            network_decode = DynamicRNNLayer(
                net_decode_in, cell_fn=cell_fn, cell_init_args=self.cell_init_args, n_hidden=n_hidden,
                initializer=initializer,
                initial_state=(network_encode.final_state if initial_state_decode is None else
                               initial_state_decode), dropout=dropout, n_layer=n_layer,
                sequence_length=decode_sequence_length, return_last=False, return_seq_2d=return_seq_2d, name='decode'
            )
            self.outputs = network_decode.outputs

            # rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # Initial state
        self.initial_state_encode = network_encode.initial_state
        self.initial_state_decode = network_decode.initial_state

        # Final state
        self.final_state_encode = network_encode.final_state
        self.final_state_decode = network_decode.final_state

        # self.sequence_length = sequence_length
        self._add_layers(network_encode.all_layers)
        self._add_params(network_encode.all_params)
        self._add_dropout_layers(network_encode.all_drop)

        self._add_layers(network_decode.all_layers)
        self._add_params(network_decode.all_params)
        self._add_dropout_layers(network_decode.all_drop)

        self._add_layers(self.outputs)
