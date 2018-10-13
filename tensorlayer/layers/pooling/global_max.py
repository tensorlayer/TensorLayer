#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['GlobalMaxPool1d', 'GlobalMaxPool2d', 'GlobalMaxPool3d']


class GlobalMaxPool1d(Layer):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = Input(x, name='in')
    >>> n = GlobalMaxPool1d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmaxpool1d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMaxPool1d, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=1, name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=2, name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )


class GlobalMaxPool2d(Layer):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

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
    >>> n = tl.layers.GlobalMaxPool2d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmaxpool2d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMaxPool2d, self).__init__()

    def __str__(self):

        additional_str = []
        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[1, 2], name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[2, 3], name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )


class GlobalMaxPool3d(Layer):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

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
    >>> n = tl.layers.GlobalMaxPool3d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name='globalmaxpool3d'):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

        self.data_format = data_format
        self.name = name

        super(GlobalMaxPool3d, self).__init__()

    def __str__(self):
        additional_str = []
        return self._str(additional_str)

    def build(self):

        if self.data_format == 'channels_last':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[1, 2, 3], name=self.name)

        elif self.data_format == 'channels_first':
            self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[2, 3, 4], name=self.name)

        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
