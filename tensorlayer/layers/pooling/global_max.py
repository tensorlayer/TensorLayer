#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['GlobalMaxPool1d', 'GlobalMaxPool2d', 'GlobalMaxPool3d']


class GlobalMaxPool1d(Layer):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = InputLayer(x, name='in')
    >>> n = GlobalMaxPool1d(n)
    [None, 30]
    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(self, prev_layer, name='globalmaxpool1d'):
        super(GlobalMaxPool1d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMaxPool1d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=1, name=name)

        self._add_layers(self._temp_data['outputs'])


class GlobalMaxPool2d(Layer):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 30])
    >>> n = tl.layers.InputLayer(x, name='in2')
    >>> n = tl.layers.GlobalMaxPool2d(n)
    [None, 30]
    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(self, prev_layer, name='globalmaxpool2d'):
        super(GlobalMaxPool2d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMaxPool2d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[1, 2], name=name)

        self._add_layers(self._temp_data['outputs'])


class GlobalMaxPool3d(Layer):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5 [batch, depth, height, width, channel].
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100, 100, 100, 30])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.GlobalMaxPool3d(n)
    [None, 30]
    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(self, prev_layer, name='globalmaxpool3d'):
        super(GlobalMaxPool3d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMaxPool3d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_max(self._temp_data['inputs'], axis=[1, 2, 3], name=name)

        self._add_layers(self._temp_data['outputs'])
