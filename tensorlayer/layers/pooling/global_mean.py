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
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3 [batch, length, channel].
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

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(self, prev_layer, name='globalmeanpool1d'):
        super(GlobalMeanPool1d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMeanPool1d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=1, name=name)


class GlobalMeanPool2d(Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

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
    >>> n = tl.layers.GlobalMeanPool2d(n)
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
    def __init__(self, prev_layer, name='globalmeanpool2d'):
        super(GlobalMeanPool2d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMeanPool2d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2], name=name)


class GlobalMeanPool3d(Layer):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

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
    >>> n = tl.layers.GlobalMeanPool2d(n)
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
    def __init__(self, prev_layer, name='globalmeanpool3d'):
        super(GlobalMeanPool3d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMeanPool3d %s" % self.name)

        self._temp_data['outputs'] = tf.reduce_mean(self._temp_data['inputs'], axis=[1, 2, 3], name=name)
