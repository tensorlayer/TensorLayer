#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Stack',
    'UnStack',
]


class Stack(Layer):
    """
    The :class:`Stack` class is a layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`__.

    Parameters
    ----------
    axis : int
        Dimension along which to concatenate.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, shape=[None, 30])
    >>> net = tl.layers.Input(x, name='input')
    >>> net1 = tl.layers.Dense(net, 10, name='dense1')
    >>> net2 = tl.layers.Dense(net, 10, name='dense2')
    >>> net3 = tl.layers.Dense(net, 10, name='dense3')
    >>> net = tl.layers.Stack([net1, net2, net3], axis=1, name='stack')
    (?, 3, 10)

    """

    def __init__(
            self,
            axis=1,
            name=None, #'stack',
    ):
        # super(Stack, self).__init__(prev_layer=layers, name=name)
        super().__init__(name)
        self.axis = axis
        logging.info("Stack %s: axis: %d" % (self.name, self.axis))

    def build(self, inputs):
        pass

    def forward(self, inputs):
        outputs = tf.stack(inputs, axis=self.axis, name=self.name)
        return outputs


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

    def __init__(self, num=None, axis=0, name=None):#'unstack'):
        # super(UnStack, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.num = num
        self.axis = axis
        logging.info("UnStack %s: num: %s axis: %d" % (self.name, self.num, self.axis))

    def build(self, inputs):
        pass

    def forward(self, inputs):
        outputs = tf.unstack(inputs, num=self.num, axis=self.axis, name=self.name)
        return outputs
