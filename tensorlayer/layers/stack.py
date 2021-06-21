#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = [
    'Stack',
    'UnStack',
]


class Stack(Module):
    """
    The :class:`Stack` class is a layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`__.

    Parameters
    ----------
    axis : int
        New dimension along which to stack.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorlayer as tl
    >>> ni = tl.layers.Input([10, 784], name='input')
    >>> net1 = tl.layers.Dense(10, name='dense1')(ni)
    >>> net2 = tl.layers.Dense(10, name='dense2')(ni)
    >>> net3 = tl.layers.Dense(10, name='dense3')(ni)
    >>> net = tl.layers.Stack(axis=1, name='stack')([net1, net2, net3])
    (10, 3, 10)

    """

    def __init__(
        self,
        axis=1,
        name=None,  #'stack',
    ):
        super().__init__(name)
        self.axis = axis

        self.build(None)
        self._built = True
        logging.info("Stack %s: axis: %d" % (self.name, self.axis))

    def __repr__(self):
        s = '{classname}(axis={axis}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        self.stack = tl.ops.Stack(axis=self.axis)

    def forward(self, inputs):
        outputs = self.stack(inputs)
        return outputs


class UnStack(Module):
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

    Examples
    --------
    >>> ni = tl.layers.Input([4, 10], name='input')
    >>> nn = tl.layers.Dense(n_units=5)(ni)
    >>> nn = tl.layers.UnStack(axis=1)(nn)  # unstack in channel axis
    >>> len(nn)  # 5
    >>> nn[0].shape  # (4,)

    """

    def __init__(self, num=None, axis=0, name=None):  #'unstack'):
        super().__init__(name)
        self.num = num
        self.axis = axis

        self.build(None)
        self._built = True
        logging.info("UnStack %s: num: %s axis: %d" % (self.name, self.num, self.axis))

    def __repr__(self):
        s = '{classname}(num={num}, axis={axis}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        self.unstack = tl.ops.Unstack(num=self.num, axis=self.axis)

    def forward(self, inputs):
        outputs = self.unstack(inputs)
        return outputs
