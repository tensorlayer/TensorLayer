#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.layers.inputs import InputLayer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
    >>> net.print_params(False)
    [TL] param   0: (100, 50)          time_dense/dense/W:0
    [TL] param   1: (50,)              time_dense/dense/b:0
    [TL]    num of params: 5050

    """

    @deprecated_alias(
        layer='prev_layer', args="layer_args", end_support_version=1.9
    )  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            layer_class=None,
            layer_args=None,
            name='time_distributed',
    ):

        super(TimeDistributedLayer, self).__init__(prev_layer=prev_layer, layer_args=layer_args, name=name)

        if not isinstance(self.inputs, tf.Tensor):
            self.inputs = tf.transpose(tf.stack(self.inputs), [1, 0, 2])

        logging.info(
            "TimeDistributedLayer %s: layer_class: %s layer_args: %s" %
            (self.name, layer_class.__name__, self.layer_args)
        )

        input_shape = self.inputs.get_shape()

        timestep = input_shape[1]
        x = tf.unstack(self.inputs, axis=1)

        is_name_reuse = tf.get_variable_scope().reuse
        for i in range(0, timestep):
            with tf.variable_scope(name, reuse=(is_name_reuse if i == 0 else True)) as vs:
                net = layer_class(InputLayer(x[i], name=self.layer_args['name'] + str(i)), **self.layer_args)
                x[i] = net.outputs
                variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.outputs = tf.stack(x, axis=1, name=name)

        self._add_layers(self.outputs)
        self._add_params(variables)
