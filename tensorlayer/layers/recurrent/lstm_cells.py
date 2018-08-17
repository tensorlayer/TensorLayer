#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer import logging

from tensorlayer.layers.recurrent.rnn_cells import ConvRNNCell


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
        self, shape, filter_size, num_features, forget_bias=1.0, input_size=None, state_is_tuple=False, act=tf.nn.tanh
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
