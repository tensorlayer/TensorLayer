#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'PoolLayer',
    'MaxPool1d',
    'MeanPool1d',
    'MaxPool2d',
    'MeanPool2d',
    'MaxPool3d',
    'MeanPool3d',
    'GlobalMaxPool1d',
    'GlobalMeanPool1d',
    'GlobalMaxPool2d',
    'GlobalMeanPool2d',
    'GlobalMaxPool3d',
    'GlobalMeanPool3d',
]


class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer.
    You can choose ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D input or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D input.

    Parameters
    ----------
    ksize : tuple of int
        The size of the window for each dimension of the input tensor.
        Note that: len(ksize) >= 4.
    strides : tuple of int
        The stride of the sliding window for each dimension of the input tensor.
        Note that: len(strides) >= 4.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    pool : pooling function
        One of ``tf.nn.max_pool``, ``tf.nn.avg_pool``, ``tf.nn.max_pool3d`` and ``f.nn.avg_pool3d``.
        See `TensorFlow pooling APIs <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`__
    name : None or str
        A unique layer name.

    Examples
    --------
    - see :class:`Conv2dLayer`.

    """

    def __init__(
            self,
            ksize=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME',
            pool=tf.nn.max_pool,
            name=None, #'pool_pro',
    ):
        # super(PoolLayer, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding
        self.pool = pool

        logging.info(
            "PoolLayer %s: ksize: %s strides: %s padding: %s pool: %s" %
            (self.name, str(self.ksize), str(self.strides), self.padding, pool.__name__)
        )

    def build(self, inputs):
        pass

    def forward(self, inputs):
        outputs = self.pool(inputs, ksize=self.ksize, strides=self.strides, padding=self.padding, name=self.name)

        return outputs


class MaxPool1d(Layer):
    """Max pooling for 1D signal.

    Parameters
    ----------
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self, filter_size=3, strides=2, padding='valid', data_format='channels_last', name=None, #'maxpool1d'
    ):
        # super(MaxPool1d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        logging.info(
            "MaxPool1d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel].
        """
        # TODO : tf.layers will be removed in TF 2.0
        outputs = tf.layers.max_pooling1d(
            inputs, self.filter_size, self.strides, padding=self.padding, data_format=self.data_format, name=self.name
        )
        return outputs



class MeanPool1d(Layer):
    """Mean pooling for 1D signal.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 3.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    """

    # logging.info("MeanPool1d %s: filter_size: %s strides: %s padding: %s" % (self.name, str(filter_size), str(strides), str(padding)))
    # outputs = tf.layers.average_pooling1d(prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)
    #
    # net_new = copy.copy(prev_layer)
    # net_new.outputs = outputs
    # net_new.all_layers.extend([outputs])
    # return net_new
    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=3, strides=2, padding='valid', data_format='channels_last', name='meanpool1d'
    ):
        super(MeanPool1d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MeanPool1d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        self.outputs = tf.layers.average_pooling1d(
            prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        self._add_layers(self.outputs)


class MaxPool2d(Layer):
    """Max pooling for 2D image.

    Parameters
    -----------
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self, filter_size=(3, 3), strides=(2, 2), padding='SAME', data_format='channels_last',
            name=None, #'maxpool2d'
    ):
        if strides is None:
            strides = filter_size

        # super(MaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size=filter_size
        self.strides=strides
        self.padding=padding
        self.data_format=data_format

        logging.info(
            "MaxPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def build(self, inputs):
        self.strides=[1, self.strides[0], self.strides[1], 1]

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4.
        """
        # outputs = tf.layers.max_pooling2d(
        #     inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        outputs = tf.nn.max_pool(inputs, ksize=self.strides, strides=self.strides, padding=self.padding, name=self.name)
        # net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1],
        #         strides=[1, strides[0], strides[1], 1],
        #         padding=padding,
        #         pool = tf.nn.max_pool,
        #         name = name)
        return outputs

class MeanPool2d(Layer):
    """Mean pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 4 [batch, height, width, channel].
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    """

    @deprecated_alias(net='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=(3, 3), strides=(2, 2), padding='SAME', data_format='channels_last',
            name='meanpool2d'
    ):

        if strides is None:
            strides = filter_size

        super(MeanPool2d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MeanPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        self.outputs = tf.layers.average_pooling2d(
            self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        self._add_layers(self.outputs)


class MaxPool3d(Layer):
    """Max pooling for 3D volume.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 3-D layer with a output rank as 5.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last',
            name='maxpool3d'
    ):
        super(MaxPool3d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MaxPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        self.outputs = tf.layers.max_pooling3d(
            self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        self._add_layers(self.outputs)


class MeanPool3d(Layer):
    """Mean pooling for 3D volume.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        The previous layer with a output rank as 5.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A mean pooling 3-D layer with a output rank as 5.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='valid', data_format='channels_last',
            name='meanpool3d'
    ):

        super(MeanPool3d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "MeanPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

        self.outputs = tf.layers.average_pooling3d(
            prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        )

        self._add_layers(self.outputs)


class GlobalMaxPool1d(Layer):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder("float32", [None, 100, 30])
    >>> n = Input(x, name='in')
    >>> n = GlobalMaxPool1d(n)
    [None, 30]
    """

    def __init__(self, data_format="channels_last", name=None):#'globalmaxpool1d'):
        # super(GlobalMaxPool1d, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("GlobalMaxPool1d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(inputs, axis=1, name=self.name)
        elif self.data_format == 'channels_first':
            self.outputs = tf.reduce_max(self.inputs, axis=2, name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class GlobalMeanPool1d(Layer):
    """The :class:`GlobalMeanPool1d` class is a 1D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
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

    def __init__(self, data_format='channels_last', name=None):#'globalmeanpool1d'):
        # super(GlobalMeanPool1d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        logging.info("GlobalMeanPool1d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(inputs, axis=1, name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(inputs, axis=2, name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class GlobalMaxPool2d(Layer):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
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

    def __init__(self, data_format='channels_last', name=None):#'globalmaxpool2d'):
        # super(GlobalMaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        logging.info("GlobalMaxPool2d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(inputs, axis=[1, 2], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_max(inputs, axis=[2, 3], name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class GlobalMeanPool2d(Layer):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
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

    def __init__(self, data_format='channels_last', name=None):#'globalmeanpool2d'):
        # super(GlobalMeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        logging.info("GlobalMeanPool2d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(inputs, axis=[1, 2], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(inputs, axis=[2, 3], name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class GlobalMaxPool3d(Layer):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
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

    def __init__(self, data_format='channels_last', name=None):#'globalmaxpool3d'):
        # super(GlobalMaxPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        logging.info("GlobalMaxPool3d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(inputs, axis=[1, 2, 3], name=self.name)
        elif data_format == 'channels_first':
            outputs = tf.reduce_max(inputs, axis=[2, 3, 4], name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class GlobalMeanPool3d(Layer):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
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

    def __init__(self, data_format='channels_last', name=None):#'globalmeanpool3d'):
        # super(GlobalMeanPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        logging.info("GlobalMeanPool3d %s" % self.name)

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(inputs, axis=[1, 2, 3], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(inputs, axis=[2, 3, 4], name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs
