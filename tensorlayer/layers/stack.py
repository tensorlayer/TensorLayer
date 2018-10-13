#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'Stack',
    'UnStack',
]


class Stack(Layer):
    """
    The :class:`Stack` class is a layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`__.

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
    >>> net = tl.layers.Input(name='input')(x)
    >>> net1 = tl.layers.Dense(10, name='dense1')(net)
    >>> net2 = tl.layers.Dense(10, name='dense2')(net)
    >>> net3 = tl.layers.Dense(10, name='dense3')(net)
    >>> net = tl.layers.Stack(axis=1, name='stack')([net1, net2, net3])
    (?, 3, 10)

    """

    def __init__(
        self,
        axis=1,
        name='stack',
    ):

        self.axis = axis
        self.name = name

        super(Stack, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("axis: %s" % self.axis)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        self._temp_data['outputs'] = tf.stack(self._temp_data['inputs'], axis=self.axis, name=self.name)


class UnStack(Layer):
    """
    The :class:`UnStack` class is a layer for unstacking the given dimension of a rank-R tensor into rank-(R-1) tensors., see `tf.unstack() <https://www.tensorflow.org/api_docs/python/tf/unstack>`__.

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

        super(UnStack, self).__init__()

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
        # https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers/stack.py#L103
        self._temp_data['outputs'] = tf.unstack(self._temp_data['inputs'], num=self.num, axis=self.axis, name=self.name)
        self.n_outputs = len(self._temp_data['outputs'])

        net_new = []

        for i, unstacked_dim in enumerate(self._temp_data['outputs']):
            layer = Layer()

            layer.name = self.name + "_%d" % i
            layer.outputs = unstacked_dim

            # TODO: CHECK THIS IMPLEMENTATION, CANNOT BE WORKING
            # need to change core layer to make this layer has all_xxx using auto-compile mode.

            layer.all_drop = self.all_drop
            layer._add_params(self.all_weights)
            layer._add_layers(self.all_layers)
            layer._add_layers(layer.outputs)

            net_new.append(layer)

        self._temp_data['outputs'] = net_new
