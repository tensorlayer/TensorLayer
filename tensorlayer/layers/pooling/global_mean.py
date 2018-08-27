#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['GlobalMeanPool1d', 'GlobalMeanPool2d', 'GlobalMeanPool3d']


class GlobalMeanPool1d(Layer):
    """The :class:`GlobalMeanPool1d` class is a 1D Global Mean Pooling layer.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.GlobalMeanPool1d(n)
    [None, 30]
    """

    def __init__(
        self,  #prev_layer,
        data_format='channels_last',
        name='globalmeanpool1d'
    ):
        self.data_format = data_format
        self.name = name
        super(GlobalMeanPool1d, self).__init__()

    def __str__(self):
        additional_str = []
        try:
            additional_str.append("output shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass
        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=1, name=self.name)
        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=2, name=self.name)
        else:
            raise Exception("data_format should be channels_last or channels_first")


class GlobalMeanPool2d(Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = tl.layers.InputLayer(x, name='in2')
    >>> n = tl.layers.GlobalMeanPool2d(n)
    [None, 30]
    """

    def __init__(
        self,  #prev_layer,
        data_format='channels_last',
        name='globalmeanpool2d'
    ):
        self.data_format = data_format
        self.name = name
        super(GlobalMeanPool2d, self).__init__()

    def __str__(self):
        additional_str = []
        try:
            additional_str.append("output shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass
        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width]
        """
        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2], name=self.name)
        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[2, 3], name=self.name)
        else:
            raise Exception("data_format should be channels_last or channels_first")


class GlobalMeanPool3d(Layer):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 100, 30])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.GlobalMeanPool2d(n)
    [None, 30]
    """

    def __init__(
        self,  #prev_layer,
        data_format='channels_last',
        name='globalmeanpool3d'
    ):
        self.data_format = data_format
        self.name = name
        super(GlobalMeanPool3d, self).__init__()

    def __str__(self):
        additional_str = []
        try:
            additional_str.append("output shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass
        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):
        """Compile.

        Parameters
        -----------
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2, 3], name=self.name)
        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[2, 3, 4], name=self.name)
        else:
            raise Exception("data_format should be channels_last or channels_first")
