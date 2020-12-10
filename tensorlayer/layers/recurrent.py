#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer

# TODO: uncomment
__all__ = [
    'RNN',
    'SimpleRNN',
    'GRURNN',
    'LSTMRNN',
    'BiRNN',
    # 'ConvRNNCell',
    # 'BasicConvLSTMCell',
    # 'ConvLSTM',
    'retrieve_seq_length_op',
    'retrieve_seq_length_op2',
    'retrieve_seq_length_op3',
    'target_mask_op',
]


class RNN(Layer):
    """
    The :class:`RNN` class is a fixed length recurrent layer for implementing simple RNN,
    LSTM, GRU and etc.

    Parameters
    ----------
    cell : TensorFlow cell function
        A RNN cell implemented by tf.keras
            - E.g. tf.keras.layers.SimpleRNNCell, tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell
            - Note TF2.0+, TF1.0+ and TF1.0- are different

    return_last_output : boolean
        Whether return last output or all outputs in a sequence.

            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False

        In a dynamic model, `return_last_output` can be updated when it is called in customised forward().
        By default, `False`.
    return_seq_2d : boolean
        Only consider this argument when `return_last_output` is `False`

            - If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
            - If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.

        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the RNN cell. The state is a list of Tensor.
        For simple RNN and GRU, last_state = [last_output]; For LSTM, last_state = [last_output, last_cell_state]

            - If True, the layer will return outputs and the final state of the cell.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.

    Examples
    --------
    For synced sequence input and output, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm.py>`__

    A simple regression model below.
    
    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out, lstm_state = tl.layers.RNN(
    >>>     cell=tf.keras.layers.LSTMCell(units=hidden_size, dropout=0.1),
    >>>     in_channels=embedding_size,
    >>>     return_last_output=True, return_last_state=True, name='lstmrnn'
    >>> )(inputs)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0], rnn_state[1]], name='rnn_model')
    >>> # If LSTMCell is applied, the rnn_state is [h, c] where h the hidden state and c the cell state of LSTM.

    A stacked RNN model.
    
    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out1 = tl.layers.RNN(
    >>>     cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     return_last_output=False, return_seq_2d=False, return_last_state=False
    >>> )(inputs)
    >>> rnn_out2 = tl.layers.RNN(
    >>>     cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     return_last_output=True, return_last_state=False
    >>> )(rnn_out1)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out2)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=outputs)

    An example if the sequences have different length and contain padding.
    Similar to the DynamicRNN in TL 1.x.

    If the `sequence_length` is provided in RNN's forwarding and both `return_last_output` and `return_last_state`
    are set as `True`, the forward function will automatically ignore the paddings. Note that if `return_last_output`
    is set as `False`, the synced sequence outputs will still include outputs which correspond with paddings,
    but users are free to select which slice of outputs to be used in following procedure.

    The `sequence_length` should be a list of integers which indicates the length of each sequence.
    It is recommended to
    `tl.layers.retrieve_seq_length_op3 <https://tensorlayer.readthedocs.io/en/latest/modules/layers.html#compute-sequence-length-3>`__
    to calculate the `sequence_length`.

    >>> data = [[[1], [2], [0], [0], [0]], [[1], [2], [3], [0], [0]], [[1], [2], [6], [1], [1]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> class DynamicRNNExample(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(DynamicRNNExample, self).__init__()
    >>>         self.rnnlayer = tl.layers.RNN(
    >>>             cell=tf.keras.layers.SimpleRNNCell(units=6, dropout=0.1), in_channels=1, return_last_output=True,
    >>>             return_last_state=True
    >>>         )
    >>>     def forward(self, x):
    >>>         z, s = self.rnnlayer(x, sequence_length=tl.layers.retrieve_seq_length_op3(x))
    >>>         return z, s
    >>> model = DynamicRNNExample()
    >>> model.eval()
    >>> output, state = model(data)


    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see layer :class:`Reshape`.

    """

    def __init__(
        self,
        cell,
        return_last_output=False,
        return_seq_2d=False,
        return_last_state=True,
        in_channels=None,
        name=None,  # 'rnn'
    ):

        super(RNN, self).__init__(name=name)

        self.cell = cell
        self.return_last_output = return_last_output
        self.return_seq_2d = return_seq_2d
        self.return_last_state = return_last_state

        if in_channels is not None:
            self.build((None, None, in_channels))
            self._built = True

        logging.info("RNN %s: cell: %s, n_units: %s" % (self.name, self.cell.__class__.__name__, self.cell.units))

    def __repr__(self):
        s = ('{classname}(cell={cellname}, n_units={n_units}')
        s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, cellname=self.cell.__class__.__name__, n_units=self.cell.units,
            **self.__dict__
        )

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor
        """
        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        if len(inputs_shape) != 3:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.name_scope(self.name) as scope:
            self.cell.build(tuple(inputs_shape))

        if self._trainable_weights is None:
            self._trainable_weights = list()
        for var in self.cell.trainable_variables:
            self._trainable_weights.append(var)

    # @tf.function
    def forward(self, inputs, sequence_length=None, initial_state=None, **kwargs):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network
        sequence_length: None or list of integers
            The actual length of each sequence in batch without padding.
            If provided, when `return_last_output` and `return_last_state` are `True`,
            the RNN will perform in the manner of a dynamic RNN, i.e.
            the RNN will return the actual last output / state without padding.
        initial_state : None or list of Tensor (RNN State)
            If None, `initial_state` is zero state.

        **kwargs: dict
            Some attributes can be updated during forwarding
            such as `return_last_output`, `return_seq_2d`, `return_last_state`.
        """
        if kwargs:
            for attr in kwargs:
                if attr in self.__dict__:
                    setattr(self, attr, kwargs[attr])

        batch_size = inputs.get_shape().as_list()[0]
        total_steps = inputs.get_shape().as_list()[1]

        # checking the type and values of sequence_length
        if sequence_length is not None:
            if isinstance(sequence_length, list):
                pass
            elif isinstance(sequence_length, tf.Tensor):
                pass
            elif isinstance(sequence_length, np.ndarray):
                sequence_length = sequence_length.tolist()
            else:
                raise TypeError(
                    "The argument sequence_length should be either None or a list of integers. "
                    "Type got %s" % type(sequence_length)
                )
            if (len(sequence_length) != batch_size):
                raise ValueError(
                    "The argument sequence_length should contain %d " % batch_size +
                    "elements indicating the initial length of each sequence, but got only %d. " % len(sequence_length)
                )
            for i in sequence_length:
                if not (type(i) is int or (isinstance(i, tf.Tensor) and i.dtype.is_integer)):
                    raise TypeError(
                        "The argument sequence_length should be either None or a list of integers. "
                        "One element of sequence_length has the type %s" % type(i)
                    )
                if i > total_steps:
                    raise ValueError(
                        "The actual length of a sequence should not be longer than "
                        "that of the longest sequence (total steps) in this mini-batch. "
                        "Total steps of this mini-batch %d, " % total_steps +
                        "but got an actual length of a sequence %d" % i
                    )

        sequence_length = tl.layers.retrieve_seq_length_op3(inputs)

        sequence_length = [i - 1 if i >= 1 else 0 for i in sequence_length]

        # set warning
        # if (not self.return_last_output) and sequence_length is not None:
        #     warnings.warn(
        #         'return_last_output is set as %s ' % self.return_last_output +
        #         'When sequence_length is provided, it is recommended to set as True. ' +
        #         'Otherwise, padding will be considered while RNN is forwarding.'
        #     )

        # return the last output, iterating each seq including padding ones. No need to store output during each
        # time step.
        if self.return_last_output and sequence_length is None:
            outputs = [-1]
        else:
            outputs = list()

        # initialize the states if provided
        states = initial_state if initial_state is not None else self.cell.get_initial_state(inputs)
        if not isinstance(states, list):
            states = [states]

        stored_states = list()

        # initialize the cell
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()

        # recurrent computation
        # FIXME: if sequence_length is provided (dynamic rnn), only iterate max(sequence_length) times.
        for time_step in range(total_steps):

            cell_output, states = self.cell.call(inputs[:, time_step, :], states, training=self.is_train)
            stored_states.append(states)

            if self.return_last_output and sequence_length is None:
                outputs[-1] = cell_output
            else:
                outputs.append(cell_output)

        # prepare to return results
        if self.return_last_output and sequence_length is None:
            outputs = outputs[-1]

        elif self.return_last_output and sequence_length is not None:
            outputs = tf.convert_to_tensor(outputs)
            outputs = tf.gather(outputs, sequence_length, axis=0)

            outputs_without_padding = []
            for i in range(batch_size):
                outputs_without_padding.append(outputs[i][i][:])
            outputs = tf.convert_to_tensor(outputs_without_padding)
        else:
            if self.return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [batch_size * n_steps, n_hidden]
                outputs = tf.reshape(tf.concat(outputs, 1), [-1, self.cell.units])
            else:
                # <akara>: stack more RNN layer after that
                # 3D Tensor [batch_size, n_steps, n_hidden]
                outputs = tf.reshape(tf.concat(outputs, 1), [-1, total_steps, self.cell.units])

        if self.return_last_state and sequence_length is None:
            return outputs, states
        elif self.return_last_state and sequence_length is not None:

            stored_states = tf.convert_to_tensor(stored_states)
            stored_states = tf.gather(stored_states, sequence_length, axis=0)

            states = []
            for i in range(stored_states.shape[1]):
                states.append(tf.convert_to_tensor([stored_states[b, i, b, :] for b in range(batch_size)]))

            return outputs, states
        else:
            return outputs


class SimpleRNN(RNN):
    """
    The :class:`SimpleRNN` class is a fixed length recurrent layer for implementing simple RNN.

    Parameters
    ----------
    units: int
        Positive integer, the dimension of hidden space.
    return_last_output : boolean
        Whether return last output or all outputs in a sequence.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False

        In a dynamic model, `return_last_output` can be updated when it is called in customised forward().
        By default, `False`.
    return_seq_2d : boolean
        Only consider this argument when `return_last_output` is `False`
            - If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
            - If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.

        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the RNN cell. The state is a list of Tensor.
        For simple RNN, last_state = [last_output]

            - If True, the layer will return outputs and the final state of the cell.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.
    `**kwargs`:
        Advanced arguments to configure the simple RNN cell.
        Please check tf.keras.layers.SimpleRNNCell.

    Examples
    --------

    A simple regression model below.

    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out, lstm_state = tl.layers.SimpleRNN(
    >>>     units=hidden_size, dropout=0.1, # both units and dropout are used to configure the simple rnn cell.
    >>>     in_channels=embedding_size,
    >>>     return_last_output=True, return_last_state=True, name='simplernn'
    >>> )(inputs)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0]], name='rnn_model')

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see layer :class:`Reshape`.

    """

    def __init__(
        self,
        units,
        return_last_output=False,
        return_seq_2d=False,
        return_last_state=True,
        in_channels=None,
        name=None,  # 'simplernn'
        **kwargs
    ):
        super(SimpleRNN, self).__init__(
            cell=tf.keras.layers.SimpleRNNCell(units=units, **kwargs), return_last_output=return_last_output,
            return_seq_2d=return_seq_2d, return_last_state=return_last_state, in_channels=in_channels, name=name
        )


class GRURNN(RNN):
    """
    The :class:`GRURNN` class is a fixed length recurrent layer for implementing RNN with GRU cell.

    Parameters
    ----------
    units: int
        Positive integer, the dimension of hidden space.
    return_last_output : boolean
        Whether return last output or all outputs in a sequence.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False

        In a dynamic model, `return_last_output` can be updated when it is called in customised forward().
        By default, `False`.
    return_seq_2d : boolean
        Only consider this argument when `return_last_output` is `False`
            - If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
            - If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.

        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the RNN cell. The state is a list of Tensor.
        For GRU, last_state = [last_output]

            - If True, the layer will return outputs and the final state of the cell.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.
    `**kwargs`:
        Advanced arguments to configure the GRU cell.
        Please check tf.keras.layers.GRUCell.

    Examples
    --------

    A simple regression model below.

    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out, lstm_state = tl.layers.GRURNN(
    >>>     units=hidden_size, dropout=0.1, # both units and dropout are used to configure the GRU cell.
    >>>     in_channels=embedding_size,
    >>>     return_last_output=True, return_last_state=True, name='grurnn'
    >>> )(inputs)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0]], name='rnn_model')

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see layer :class:`Reshape`.

    """

    def __init__(
        self,
        units,
        return_last_output=False,
        return_seq_2d=False,
        return_last_state=True,
        in_channels=None,
        name=None,  # 'grurnn'
        **kwargs
    ):
        super(GRURNN, self).__init__(
            cell=tf.keras.layers.GRUCell(units=units, **kwargs), return_last_output=return_last_output,
            return_seq_2d=return_seq_2d, return_last_state=return_last_state, in_channels=in_channels, name=name
        )


class LSTMRNN(RNN):
    """
    The :class:`LSTMRNN` class is a fixed length recurrent layer for implementing RNN with LSTM cell.

    Parameters
    ----------
    units: int
        Positive integer, the dimension of hidden space.
    return_last_output : boolean
        Whether return last output or all outputs in a sequence.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False

        In a dynamic model, `return_last_output` can be updated when it is called in customised forward().
        By default, `False`.
    return_seq_2d : boolean
        Only consider this argument when `return_last_output` is `False`
            - If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
            - If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.

        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the RNN cell. The state is a list of Tensor.
        For LSTM, last_state = [last_output, last_cell_state]

            - If True, the layer will return outputs and the final state of the cell.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.
    `**kwargs`:
        Advanced arguments to configure the LSTM cell.
        Please check tf.keras.layers.LSTMCell.

    Examples
    --------

    A simple regression model below.

    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out, lstm_state = tl.layers.LSTMRNN(
    >>>     units=hidden_size, dropout=0.1, # both units and dropout are used to configure the LSTM cell.
    >>>     in_channels=embedding_size,
    >>>     return_last_output=True, return_last_state=True, name='grurnn'
    >>> )(inputs)
    >>> outputs = tl.layers.Dense(n_units=1)(rnn_out)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0]], name='rnn_model')

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see layer :class:`Reshape`.

    """

    def __init__(
        self,
        units,
        return_last_output=False,
        return_seq_2d=False,
        return_last_state=True,
        in_channels=None,
        name=None,  # 'lstmrnn'
        **kwargs
    ):
        super(LSTMRNN, self).__init__(
            cell=tf.keras.layers.LSTMCell(units=units, **kwargs), return_last_output=return_last_output,
            return_seq_2d=return_seq_2d, return_last_state=return_last_state, in_channels=in_channels, name=name
        )


class BiRNN(Layer):
    """
    The :class:`BiRNN` class is a fixed length Bidirectional recurrent layer.

    Parameters
    ----------
    fw_cell : TensorFlow cell function for forward direction
        A RNN cell implemented by tf.keras, e.g. tf.keras.layers.SimpleRNNCell, tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell.
        Note TF2.0+, TF1.0+ and TF1.0- are different
    bw_cell: TensorFlow cell function for backward direction similar with `fw_cell`
    return_seq_2d : boolean.
        If True, return 2D Tensor [batch_size * n_steps, n_hidden], for stacking Dense layer after it.
        If False, return 3D Tensor [batch_size, n_steps, n_hidden], for stacking multiple RNN after it.
        In a dynamic model, `return_seq_2d` can be updated when it is called in customised forward().
        By default, `False`.
    return_last_state: boolean
        Whether to return the last state of the two cells. The state is a list of Tensor.
            - If True, the layer will return outputs, the final state of `fw_cell` and the final state of `bw_cell`.
            - If False, the layer will return outputs only.

        In a dynamic model, `return_last_state` can be updated when it is called in customised forward().
        By default, `False`.
    in_channels: int
        Optional, the number of channels of the previous layer which is normally the size of embedding.
        If given, the layer will be built when init.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : str
        A unique layer name.

    Examples
    --------
    A simple regression model below.
    
    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> # the fw_cell and bw_cell can be different
    >>> rnnlayer = tl.layers.BiRNN(
    >>>     fw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     bw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size + 1, dropout=0.1),
    >>>     return_seq_2d=True, return_last_state=True
    >>> )
    >>> # if return_last_state=True, the final state of the two cells will be returned together with the outputs
    >>> # if return_last_state=False, only the outputs will be returned
    >>> rnn_out, rnn_fw_state, rnn_bw_state = rnnlayer(inputs)
    >>> # if the BiRNN is followed by a Dense, return_seq_2d should be True.
    >>> # if the BiRNN is followed by other RNN, return_seq_2d can be False.
    >>> dense = tl.layers.Dense(n_units=1)(rnn_out)
    >>> outputs = tl.layers.Reshape([-1, num_steps])(dense)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_out, rnn_fw_state[0], rnn_bw_state[0]])

    A stacked BiRNN model.
    
    >>> inputs = tl.layers.Input([batch_size, num_steps, embedding_size])
    >>> rnn_out1 = tl.layers.BiRNN(
    >>>     fw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     bw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size + 1, dropout=0.1),
    >>>     return_seq_2d=False, return_last_state=False
    >>> )(inputs)
    >>> rnn_out2 = tl.layers.BiRNN(
    >>>     fw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size, dropout=0.1),
    >>>     bw_cell=tf.keras.layers.SimpleRNNCell(units=hidden_size + 1, dropout=0.1),
    >>>     return_seq_2d=True, return_last_state=False
    >>> )(rnn_out1)
    >>> dense = tl.layers.Dense(n_units=1)(rnn_out2)
    >>> outputs = tl.layers.Reshape([-1, num_steps])(dense)
    >>> rnn_model = tl.models.Model(inputs=inputs, outputs=outputs)

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features]. If not, please see layer :class:`Reshape`.

    """

    def __init__(
        self,
        fw_cell,
        bw_cell,
        return_seq_2d=False,
        return_last_state=False,
        in_channels=None,
        name=None,  # 'birnn'
    ):
        super(BiRNN, self).__init__(name)

        self.fw_cell = fw_cell
        self.bw_cell = bw_cell
        self.return_seq_2d = return_seq_2d
        self.return_last_state = return_last_state

        if in_channels is not None:
            self.build((None, None, in_channels))
            self._built = True

        logging.info(
            "BiRNN %s: fw_cell: %s, fw_n_units: %s, bw_cell: %s, bw_n_units： %s" % (
                self.name, self.fw_cell.__class__.__name__, self.fw_cell.units, self.bw_cell.__class__.__name__,
                self.bw_cell.units
            )
        )

    def __repr__(self):
        s = (
            '{classname}(fw_cell={fw_cellname}, fw_n_units={fw_n_units}'
            ', bw_cell={bw_cellname}, bw_n_units={bw_n_units}'
        )
        s += ', name=\'{name}\''
        s += ')'
        return s.format(
            classname=self.__class__.__name__, fw_cellname=self.fw_cell.__class__.__name__,
            fw_n_units=self.fw_cell.units, bw_cellname=self.bw_cell.__class__.__name__, bw_n_units=self.bw_cell.units,
            **self.__dict__
        )

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor
        """
        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        if len(inputs_shape) != 3:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.name_scope(self.name) as scope:
            self.fw_cell.build(tuple(inputs_shape))
            self.bw_cell.build(tuple(inputs_shape))

        if self._trainable_weights is None:
            self._trainable_weights = list()
        for var in self.fw_cell.trainable_variables:
            self._trainable_weights.append(var)
        for var in self.bw_cell.trainable_variables:
            self._trainable_weights.append(var)

    # @tf.function
    def forward(self, inputs, fw_initial_state=None, bw_initial_state=None, **kwargs):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network
        fw_initial_state : None or list of Tensor (RNN State)
            If None, `fw_initial_state` is zero state.
        bw_initial_state : None or list of Tensor (RNN State)
            If None, `bw_initial_state` is zero state.
        **kwargs: dict
            Some attributes can be updated during forwarding
            such as `return_last_output`, `return_seq_2d`, `return_last_state`.
        """

        if kwargs:
            for attr in kwargs:
                if attr in self.__dict__:
                    setattr(self, attr, kwargs[attr])

        fw_outputs = list()
        bw_outputs = list()

        fw_states = fw_initial_state if fw_initial_state is not None else self.fw_cell.get_initial_state(inputs)
        bw_states = bw_initial_state if bw_initial_state is not None else self.bw_cell.get_initial_state(inputs)

        if not isinstance(fw_states, list):
            fw_states = [fw_states]
        if not isinstance(bw_states, list):
            bw_states = [bw_states]

        total_steps = inputs.get_shape().as_list()[1]

        self.fw_cell.reset_dropout_mask()
        self.fw_cell.reset_recurrent_dropout_mask()
        self.bw_cell.reset_dropout_mask()
        self.bw_cell.reset_recurrent_dropout_mask()

        for time_step in range(total_steps):

            fw_cell_output, fw_states = self.fw_cell.call(inputs[:, time_step, :], fw_states, training=self.is_train)
            bw_cell_output, bw_states = self.bw_cell.call(
                inputs[:, -time_step - 1, :], bw_states, training=self.is_train
            )

            fw_outputs.append(fw_cell_output)
            bw_outputs.append(bw_cell_output)

        if self.return_seq_2d:
            # PTB tutorial: stack dense layer after that, or compute the cost from the output
            # 2D Tensor [batch_size * n_steps, n_hidden]
            fw_outputs = tf.reshape(tf.concat(fw_outputs, 1), [-1, self.fw_cell.units])
            bw_outputs = tf.reshape(tf.concat(bw_outputs, 1), [-1, self.bw_cell.units])
        else:
            # <akara>: stack more RNN layer after that
            # 3D Tensor [batch_size, n_steps, n_hidden]
            fw_outputs = tf.reshape(tf.concat(fw_outputs, 1), [-1, total_steps, self.fw_cell.units])
            bw_outputs = tf.reshape(tf.concat(bw_outputs, 1), [-1, total_steps, self.bw_cell.units])

        outputs = tf.concat([fw_outputs, bw_outputs], -1)

        if self.return_last_state:
            return outputs, fw_states, bw_states
        else:
            return outputs


'''
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

    def zero_state(self, batch_size):  #, dtype=LayersConfig.tf_dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros

        """
        dtype = LayersConfig.tf_dtype
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
        with tf.compat.v1.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
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
    with tf.compat.v1.variable_scope(scope or "Conv"):
        matrix = tf.compat.v1.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype
        )
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(args, 3), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.compat.v1.get_variable(
            "Bias", [num_features], dtype=dtype,
            initializer=tf.compat.v1.initializers.constant(bias_start, dtype=dtype)
        )
    return res + bias_term


class ConvLSTM(Layer):
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
        Only consider this argument when `return_last_output` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.

    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : tensor
        The output of this RNN. return_last_output = False, outputs = all cell_output, which is the hidden state.
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
            initializer=tf.compat.v1.initializers.random_uniform(-0.1, 0.1),
            n_steps=5,
            initial_state=None,
            return_last=False,
            return_seq_2d=False,
            name='convlstm',
    ):
        super(ConvLSTM, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "ConvLSTM %s: feature_map: %d, n_steps: %d, "
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
        with tf.compat.v1.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.compat.v1.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :, :, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.VARIABLES, scope=vs.name)

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

'''


# @tf.function
def retrieve_seq_length_op(data):
    """An op to compute the length of a sequence from input shape of [batch_size, n_step(max), n_features],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max), n_features] with zero padding on right hand side.

    Examples
    -----------
    Single feature

    >>> data = [[[1],[2],[0],[0],[0]],
    >>>         [[1],[2],[3],[0],[0]],
    >>>         [[1],[2],[6],[1],[0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op(data)
    [2 3 4]

    Multiple features

    >>> data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
    >>>          [[2,3],[2,4],[3,2],[0,0],[0,0]],
    >>>          [[3,3],[2,2],[5,3],[1,2],[0,0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op(data)
    [4 3 4]

    References
    ------------
    Borrow from `TFlearn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__.

    """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(input_tensor=tf.abs(data), axis=2))
        length = tf.reduce_sum(input_tensor=used, axis=1)

        return tf.cast(length, tf.int32)


# @tf.function
def retrieve_seq_length_op2(data):
    """An op to compute the length of a sequence, from input shape of [batch_size, n_step(max)],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] with zero padding on right hand side.

    Examples
    -----------
    >>> data = [[1,2,0,0,0],
    >>>         [1,2,3,0,0],
    >>>         [1,2,6,1,0]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op2(data)
    tensor([2 3 4])

    """
    return tf.reduce_sum(input_tensor=tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), axis=1)


# @tf.function
def retrieve_seq_length_op3(data, pad_val=0):
    """An op to compute the length of a sequence, the data shape can be [batch_size, n_step(max)] or
    [batch_size, n_step(max), n_features].

    If the data has type of tf.string and pad_val is assigned as empty string (''), this op will compute the
    length of the string sequence.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] or [batch_size, n_step(max), n_features] with zero padding on the right hand side.
    pad_val:
        By default 0. If the data is tf.string, please assign this as empty string ('')

    Examples
    -----------
    >>> data = [[[1],[2],[0],[0],[0]],
    >>>         [[1],[2],[3],[0],[0]],
    >>>         [[1],[2],[6],[1],[0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op3(data)
    tensor([2, 3, 4])
    >>> data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
    >>>         [[2,3],[2,4],[3,2],[0,0],[0,0]],
    >>>         [[3,3],[2,2],[5,3],[1,2],[0,0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op3(data)
    tensor([4, 3, 4])
    >>> data = [[1,2,0,0,0],
    >>>         [1,2,3,0,0],
    >>>         [1,2,6,1,0]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> length = tl.layers.retrieve_seq_length_op3(data)
    tensor([2, 3, 4])
    >>> data = [['hello','world','','',''],
    >>>         ['hello','world','tensorlayer','',''],
    >>>         ['hello','world','tensorlayer','2.0','']]
    >>> data = tf.convert_to_tensor(data, dtype=tf.string)
    >>> length = tl.layers.retrieve_seq_length_op3(data, pad_val='')
    tensor([2, 3, 4])

    """
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.reduce_sum(
            input_tensor=tf.cast(tf.reduce_any(input_tensor=tf.not_equal(data, pad_val), axis=2), dtype=tf.int32),
            axis=1
        )
    elif data_shape_size == 2:
        return tf.reduce_sum(input_tensor=tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32), axis=1)
    elif data_shape_size == 1:
        raise ValueError("retrieve_seq_length_op3: data has wrong shape! Shape got ", data.get_shape().as_list())
    else:
        raise ValueError(
            "retrieve_seq_length_op3: handling data with num of dims %s hasn't been implemented!" % (data_shape_size)
        )


def target_mask_op(data, pad_val=0):
    """ Return the mask of the input sequence data based on the padding values.

    Parameters
    -----------
    data : tf.Tensor
        A tensor with 2 or 3 dimensions.
    pad_val: int, float, string, etc
        The value that represent padding. By default, 0. For tf.string, you may use empty string.

    Examples
    -----------
    >>> data = [['hello', 'world', '', '', ''],
    >>>         ['hello', 'world', 'tensorlayer', '', ''],
    >>>         ['hello', 'world', 'tensorlayer', '2.0', '']]
    >>> data = tf.convert_to_tensor(data, dtype=tf.string)
    >>> mask = tl.layers.target_mask_op(data, pad_val='')
    >>> print(mask)
    tf.Tensor(
    [[1 1 0 0 0]
     [1 1 1 0 0]
     [1 1 1 1 0]], shape=(3, 5), dtype=int32)
    >>> data = [[[1], [0], [0], [0], [0]],
    >>>         [[1], [2], [3], [0], [0]],
    >>>         [[1], [2], [0], [1], [0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> mask = tl.layers.target_mask_op(data)
    >>> print(mask)
    tf.Tensor(
    [[1 0 0 0 0]
     [1 1 1 0 0]
     [1 1 0 1 0]], shape=(3, 5), dtype=int32)
    >>> data = [[[0,0],[2,2],[1,2],[1,2],[0,0]],
    >>>         [[2,3],[2,4],[3,2],[1,0],[0,0]],
    >>>         [[3,3],[0,1],[5,3],[1,2],[0,0]]]
    >>> data = tf.convert_to_tensor(data, dtype=tf.float32)
    >>> mask = tl.layers.target_mask_op(data)
    >>> print(mask)
    tf.Tensor(
    [[0 1 1 1 0]
     [1 1 1 1 0]
     [1 1 1 1 0]], shape=(3, 5), dtype=int32)
    """

    if not isinstance(data, tf.Tensor):
        raise AttributeError("target_mask_op: the type of input data should be tf.Tensor but got %s." % type(data))
    data_shape_size = data.get_shape().ndims
    if data_shape_size == 3:
        return tf.cast(tf.reduce_any(input_tensor=tf.not_equal(data, pad_val), axis=2), dtype=tf.int32)
    elif data_shape_size == 2:
        return tf.cast(tf.not_equal(data, pad_val), dtype=tf.int32)
    elif data_shape_size == 1:
        raise ValueError(
            "target_mask_op: data_shape %s is not supported. "
            "The shape of data should have 2 or 3 dims." % (data.get_shape())
        )
    else:
        raise ValueError(
            "target_mask_op: handling data_shape %s hasn't been implemented! "
            "The shape of data should have 2 or 3 dims" % (data.get_shape())
        )
