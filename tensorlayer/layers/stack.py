#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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
        axis=1,
        name='stack',
    ):

        self.axis = axis
        self.name = name

        super(StackLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("axis: %s" % self.axis)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        self._temp_data['outputs'] = tf.stack(self._temp_data['inputs'], axis=self.axis, name=self.name)


class UnStackLayer(Layer):
    """
    The :class:`UnStackLayer` class is a layer for unstacking the given dimension of a rank-R tensor into rank-(R-1) tensors., see `tf.unstack() <https://www.tensorflow.org/api_docs/python/tf/unstack>`__.

    Parameters
    ----------
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

    def __init__(self, num=None, axis=0, name='unstack'):

        self.num = num
        self.axis = axis
        self.name = name

        super(UnStackLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("num: %s" % self.num)
        except AttributeError:
            pass

        try:
            additional_str.append("axis: %s" % self.axis)
        except AttributeError:
            pass

        try:
            additional_str.append("n_outputs: %s" % self.n_outputs)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = tf.unstack(self._temp_data['inputs'], num=self.num, axis=self.axis, name=self.name)
        self.n_outputs = len(self._temp_data['outputs'])

        net_new = []

        for i, unstacked_dim in enumerate(self._temp_data['outputs']):
            layer = Layer()

            layer.name = self.name + "_%d" % i
            layer.outputs = unstacked_dim

            # TODO: CHECK THIS IMPLEMENTATION, CANNOT BE WORKING

            layer.all_drop = self.all_drop
            layer._add_params(self.all_weights)
            layer._add_layers(self.all_layers)
            layer._add_layers(layer.outputs)

            net_new.append(layer)

        self._temp_data['outputs'] = net_new
