#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging

from tensorlayer.layers.core import Layer

__all__ = [
    'Multiplexer',
]


class Multiplexer(Layer):
    """
    The :class:`Multiplexer` selects inputs to be forwarded to output.
    see `tutorial_mnist_multiplexer.py`.

    Parameters
    ----------
    layers : a list of :class:`Layer`
        The input layers.
    name : str
        A unique layer name.

    Attributes
    ----------
    sel : placeholder
        The placeholder takes an integer for selecting which layer to output.

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=(None, 784), name='x')
    >>> # define the network
    >>> net_in = tl.layers.Input(x, name='input')
    >>> net_in = tl.layers.Dropout(net_in, keep=0.8, name='drop1')
    >>> # net 0
    >>> net_0 = tl.layers.Dense(net_in, n_units=800, act=tf.nn.relu, name='net0/relu1')
    >>> net_0 = tl.layers.Dropout(net_0, keep=0.5, name='net0/drop2')
    >>> net_0 = tl.layers.Dense(net_0, n_units=800, act=tf.nn.relu, name='net0/relu2')
    >>> # net 1
    >>> net_1 = tl.layers.Dense(net_in, n_units=800, act=tf.nn.relu, name='net1/relu1')
    >>> net_1 = tl.layers.Dropout(net_1, keep=0.8, name='net1/drop2')
    >>> net_1 = tl.layers.Dense(net_1, n_units=800, act=tf.nn.relu, name='net1/relu2')
    >>> net_1 = tl.layers.Dropout(net_1, keep=0.8, name='net1/drop3')
    >>> net_1 = tl.layers.Dense(net_1, n_units=800, act=tf.nn.relu, name='net1/relu3')
    >>> # multiplexer
    >>> net_mux = tl.layers.Multiplexer(layers=[net_0, net_1], name='mux')
    >>> network = tl.layers.Reshape(net_mux, shape=(-1, 800), name='reshape')
    >>> network = tl.layers.Dropout(network, keep=0.5, name='drop3')
    >>> # output layer
    >>> network = tl.layers.Dense(network, n_units=10, act=None, name='output')

    """

    def __init__(self, layers, name='mux'):

        super(Multiplexer, self).__init__(prev_layer=layers, name=name)

        self.n_inputs = len(layers)

        all_inputs = tf.stack(self.inputs, name=name)  # pack means concat a list of tensor in a new dim  # 1.2

        logging.info("Multiplexer %s: n_inputs: %d" % (self.name, self.n_inputs))

        self.sel = tf.placeholder(tf.int32)
        self.outputs = tf.gather(all_inputs, self.sel, name=name)  # [sel, :, : ...] # 1.2

        # logging.info(self.outputs, vars(self.outputs))
        #         # tf.reshape(self.outputs, shape=)
        # exit()
        # # the same with ConcatLayer

        # for i in range(1, len(layers)):
        #     self._add_layers(list(layers[i].all_layers))
        #     self._add_params(list(layers[i].all_params))
        #     self.all_drop.update(dict(layers[i].all_drop))

        self._add_layers(self.outputs)
