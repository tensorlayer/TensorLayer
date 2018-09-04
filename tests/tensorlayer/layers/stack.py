#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'StackLayer',
    'UnStackLayer',
]


class StackLayer(Layer):
    """
    The :class:`StackLayer` class is a layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`__.

    Parameters
    ----------
    layers : list of :class:`Layer`
        Previous layers to stack.
    axis : int
        Dimension along which to concatenate.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 30])
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net1 = tl.layers.DenseLayer(net, 10, name='dense1')
    >>> net2 = tl.layers.DenseLayer(net, 10, name='dense2')
    >>> net3 = tl.layers.DenseLayer(net, 10, name='dense3')
    >>> net = tl.layers.StackLayer([net1, net2, net3], axis=1, name='stack')
    (?, 3, 10)

    """

    def __init__(
            self,
            layers,
            axis=1,
            name='stack',
    ):

        super(StackLayer, self).__init__(prev_layer=layers, name=name)

        logging.info("StackLayer %s: axis: %d" % (self.name, axis))

        self.outputs = tf.stack(self.inputs, axis=axis, name=name)

        # for i in range(1, len(layers)):
        #     self._add_layers(list(layers[i].all_layers))
        #     self._add_params(list(layers[i].all_params))
        #     self.all_drop.update(dict(layers[i].all_drop))

        self._add_layers(self.outputs)


class UnStackLayer(Layer):
    """
    The :class:`UnStackLayer` class is a layer for unstacking the given dimension of a rank-R tensor into rank-(R-1) tensors., see `tf.unstack() <https://www.tensorflow.org/api_docs/python/tf/unstack>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    num : int or None
        The length of the dimension axis. Automatically inferred if None (the default).
    axis : int
        Dimension along which axis to concatenate.
    name : str
        A unique layer name.

    Returns
    -------
    list of :class:`Layer`
        The list of layer objects unstacked from the input.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, num=None, axis=0, name='unstack'):

        super(UnStackLayer, self).__init__(prev_layer=prev_layer, name=name)

        outputs = tf.unstack(self.inputs, num=num, axis=axis, name=name)

        logging.info("UnStackLayer %s: num: %s axis: %d, n_outputs: %d" % (self.name, num, axis, len(outputs)))

        net_new = []

        for i, unstacked_dim in enumerate(outputs):
            layer = Layer(prev_layer=self, name=name + str(i))
            layer.outputs = unstacked_dim

            net_new.append(layer)

        self.outputs = net_new

        self._add_layers(net_new)
