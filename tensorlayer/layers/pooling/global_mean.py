#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['GlobalMeanPool1d', 'GlobalMeanPool2d', 'GlobalMeanPool3d']


class GlobalMeanPool1d(Layer):
    """The :class:`GlobalMeanPool1d` class is a 1D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = tl.layers.Input(x, name='in')
    >>> n = tl.layers.GlobalMeanPool1d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmeanpool1d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMeanPool1d, self).__init__()

    def __str__(self):
        additional_str = []

        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=1, name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=2, name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )


class GlobalMeanPool2d(Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = tl.layers.Input(x, name='in2')
    >>> n = tl.layers.GlobalMeanPool2d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmeanpool2d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMeanPool2d, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2], name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[2, 3], name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )


class GlobalMeanPool3d(Layer):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 100, 30])
    >>> n = tl.layers.Input(x, name='in')
    >>> n = tl.layers.GlobalMeanPool2d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmeanpool3d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMeanPool3d, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2, 3], name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[2, 3, 4], name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
