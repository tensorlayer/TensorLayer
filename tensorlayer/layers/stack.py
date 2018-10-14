#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import BuiltLayer

from tensorlayer.decorators import private_method

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
            additional_str.append("n_outputs: %d" % len(self._temp_data['outputs']))
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        # https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers/stack.py#L103

        unstacked_layers = tf.unstack(self._temp_data['inputs'], num=self.num, axis=self.axis, name=self.name)

        self._temp_data['outputs'] = []

        for i, unstacked_dim in enumerate(unstacked_layers):

            self._temp_data['outputs'].append(
                self._create_unstacked_layer(name=self.name + "_%d" % (i + 1), outputs=unstacked_dim)
            )

        self.parse_outputs(self._temp_data['outputs'])

    @private_method
    def _create_unstacked_layer(self, name, outputs):

        _str_ = "UnStackedLayer: %s - output shape: %s" % (name, outputs.shape)

        return type("Built_UnStackedLayer", (BuiltLayer, ), {})(
            layers_to_build=None,
            inputs=self,
            outputs=outputs,
            local_weights=list(),
            local_drop=list(),
            is_train=self._temp_data['is_train'],
            name=name,
            _str_=_str_
        )

    @private_method
    def parse_outputs(self, outputs):

        class UnStackArray(object):

            def __init__(self, ndarr):
                self.ndarr = ndarr

            def __getattribute__(self, item):

                if item == "dtype":
                    return self.ndarr[0].outputs.dtype

                elif item == "shape":

                    def parse_dim(dim):
                        try:
                            return int(dim)
                        except TypeError:
                            return None

                    return tuple([parse_dim(i) for i in [len(self.ndarr)] + list(self.ndarr[0].outputs.shape)])

                else:
                    return super(UnStackArray, self).__getattribute__(item)

            def __getitem__(self, i):
                return self.ndarr[i]

            def __len__(self):
                return len(self.ndarr)

        self._temp_data['outputs'] = UnStackArray(np.array(outputs))
