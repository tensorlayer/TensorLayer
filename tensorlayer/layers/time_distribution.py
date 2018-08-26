#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.layers.inputs import InputLayer

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'TimeDistributedLayer',
]


class TimeDistributedLayer(Layer):
    """
    The :class:`TimeDistributedLayer` class that applies a function to every timestep of the input tensor.
    For example, if use :class:`DenseLayer` as the `layer_class`, we input (batch_size, length, dim) and
    output (batch_size , length, new_dim).

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with output size of (batch_size, length, dim).
    layer_class : a :class:`Layer` class
        The layer class name.
    args : dictionary
        The arguments for the ``layer_class``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 32
    >>> timestep = 20
    >>> input_dim = 100
    >>> x = tf.placeholder(dtype=tf.float32, shape=[batch_size, timestep, input_dim], name="encode_seqs")
    >>> net = tl.layers.InputLayer(x, name='input')
    [TL] InputLayer  input: (32, 20, 100)
    >>> net = tl.layers.TimeDistributedLayer(net, layer_class=tl.layers.DenseLayer, args={'n_units':50, 'name':'dense'}, name='time_dense')
    [TL] TimeDistributedLayer time_dense: layer_class:DenseLayer
    >>> print(net.outputs._shape)
    (32, 20, 50)
    >>> net.print_weights(False)
    [TL] param   0: (100, 50)          time_dense/dense/W:0
    [TL] param   1: (50,)              time_dense/dense/b:0
    [TL]    num of params: 5050

    """

    @deprecated_alias(
        layer='prev_layer', args="layer_args", end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        layer_class=None,
        layer_args=None,
        name='time_distributed',
    ):

        self.prev_layer = prev_layer
        self.layer_class = layer_class
        self.name = name

        super(TimeDistributedLayer, self).__init__(layer_args=layer_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer_class: %s" % self.layer_class.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("layer_args: %s" % self.layer_args)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        if not isinstance(self._temp_data['inputs'], tf.Tensor):
            self._temp_data['inputs'] = tf.transpose(tf.stack(self._temp_data['inputs']), [1, 0, 2])

        input_shape = self._temp_data['inputs'].get_shape()

        timestep = input_shape[1]

        x = tf.unstack(self._temp_data['inputs'], axis=1)

        is_name_reuse = tf.get_variable_scope().reuse

        for i in range(0, timestep):

            reuse = is_name_reuse if i == 0 else True
            with tf.variable_scope(self.name, reuse=reuse) as vs:

                in_layer = InputLayer(x[i], name=self.layer_args['name'] + str(i))

                net = self.layer_class(in_layer, **self.layer_args)
                x[i] = net.outputs

                self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._temp_data['outputs'] = tf.stack(x, axis=1, name=self.name)
