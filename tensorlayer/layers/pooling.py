#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

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
    'AdaptiveMeanPool1d',
    'AdaptiveMeanPool2d',
    'AdaptiveMeanPool3d',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
    'CornerPool2d',
]


class PoolLayer(Module):
    """
    The :class:`PoolLayer` class is a Pooling layer.
    You can choose ``tl.ops.max_pool`` and ``tl.ops.avg_pool`` for 2D input or
    ``tl.ops.max_pool3d`` and ``tl.ops.avg_pool3d`` for 3D input.

    Parameters
    ----------
    filter_size : tuple of int
        The size of the window for each dimension of the input tensor.
        Note that: len(filter_size) >= 4.
    strides : tuple of int
        The stride of the sliding window for each dimension of the input tensor.
        Note that: len(strides) >= 4.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    pool : pooling function
        One of ``tl.ops.max_pool``, ``tl.ops.avg_pool``, ``tl.ops.max_pool3d`` and ``f.ops.avg_pool3d``.
        See `TensorFlow pooling APIs <https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/>`__
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tl.layers.PoolLayer()(net)
    >>> output shape : [None, 25, 25, 32]

    """

    def __init__(
        self,
        filter_size=(1, 2, 2, 1),
        strides=(1, 2, 2, 1),
        padding='SAME',
        pool=tl.ops.MaxPool,
        name=None  # 'pool_pro',
    ):
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.pool = pool

        self.build()
        self._built = True

        logging.info(
            "PoolLayer %s: filter_size: %s strides: %s padding: %s pool: %s" %
            (self.name, str(self.filter_size), str(self.strides), self.padding, pool.__name__)
        )

    def __repr__(self):
        s = '{classname}(pool={poolname}, filter_size={strides}, padding={padding}'
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, poolname=self.pool.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self._pool = self.pool(ksize=self.filter_size, strides=self.strides, padding=self.padding)

    def forward(self, inputs):
        outputs = self._pool(inputs)
        return outputs


class MaxPool1d(Module):
    """Max pooling for 1D signal.

    Parameters
    ----------
    filter_size : int
        Pooling window size.
    strides : int
        Stride of the pooling operation.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 32], name='input')
    >>> net = tl.layers.MaxPool1d(filter_size=3, strides=2, padding='SAME', name='maxpool1d')(net)
    >>> output shape : [None, 25, 32]

    """

    def __init__(
        self,
        filter_size=3,
        strides=2,
        padding='SAME',
        data_format='channels_last',
        name=None  # 'maxpool1d'
    ):
        super().__init__(name)
        self.filter_size = self._filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool1d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/pool
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")
        self._filter_size = [self.filter_size]
        self._strides = [self.strides]
        self.max_pool = tl.ops.MaxPool1d(ksize=self._filter_size, strides=self._strides, padding=self.padding,
                                       data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.max_pool(inputs)
        return outputs

class MeanPool1d(Module):
    """Mean pooling for 1D signal.

    Parameters
    ------------
    filter_size : int
        Pooling window size.
    strides : int
        Strides of the pooling operation.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 32], name='input')
    >>> net = tl.layers.MeanPool1d(filter_size=3, strides=2, padding='SAME')(net)
    >>> output shape : [None, 25, 32]

    """

    def __init__(
        self,
        filter_size=3,
        strides=2,
        padding='SAME',
        data_format='channels_last',
        dilation_rate=1,
        name=None  # 'meanpool1d'
    ):
        super().__init__(name)
        self.filter_size = self._filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MeanPool1d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn/pool
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")
        self._filter_size = [self.filter_size]
        self._strides = [self.strides]
        self.avg_pool = tl.ops.AvgPool1d(ksize=self._filter_size,
                                         strides=self._strides,
                                         padding=self.padding,
                                         data_format=self.data_format)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        return outputs


class MaxPool2d(Module):
    """Max pooling for 2D image.

    Parameters
    -----------
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tl.layers.MaxPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME')(net)
    >>> output shape : [None, 25, 25, 32]

    """

    def __init__(
        self,
        filter_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        data_format='channels_last',
        name=None  # 'maxpool2d'
    ):
        super().__init__(name)
        self.filter_size = filter_size
        if strides is None:
            strides = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            self._strides = [1, self.strides[0], self.strides[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            self._strides = [1, 1, self.strides[0], self.strides[1]]
        else:
            raise Exception("unsupported data format")

        self.max_pool = tl.ops.MaxPool(
            ksize=self.filter_size, strides=self._strides, padding=self.padding, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.max_pool(inputs)
        return outputs


class MeanPool2d(Module):
    """Mean pooling for 2D image [batch, height, width, channel].

    Parameters
    -----------
    filter_size : tuple of int
        (height, width) for filter size.
    strides : tuple of int
        (height, width) for strides.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 32], name='input')
    >>> net = tl.layers.MeanPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME')(net)
    >>> output shape : [None, 25, 25, 32]

    """

    def __init__(
        self,
        filter_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        data_format='channels_last',
        name=None  # 'meanpool2d'
    ):
        super().__init__(name)
        self.filter_size = filter_size
        if strides is None:
            strides = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MeanPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
            self._strides = [1, self.strides[0], self.strides[1], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
            self._strides = [1, 1, self.strides[0], self.strides[1]]
        else:
            raise Exception("unsupported data format")
        self.avg_pool = tl.ops.AvgPool(
            ksize=self.filter_size, strides=self._strides, padding=self.padding, data_format=self.data_format
        )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        return outputs


class MaxPool3d(Module):
    """Max pooling for 3D volume.

    Parameters
    ------------
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`tf.Tensor`
        A max pooling 3-D layer with a output rank as 5.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 50, 32], name='input')
    >>> net = tl.layers.MaxPool3d(filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(net)
    >>> output shape : [None, 25, 25, 25, 32]

    """

    def __init__(
        self,
        filter_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding='VALID',
        data_format='channels_last',
        name=None  # 'maxpool3d'
    ):
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
            self._strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
            self._strides = [1, 1, self.strides[0], self.strides[1], self.strides[2]]
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        outputs = tl.ops.max_pool3d(
            input=inputs,
            ksize=self.filter_size,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


class MeanPool3d(Module):
    """Mean pooling for 3D volume.

    Parameters
    ------------
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'VALID' or 'SAME'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`tf.Tensor`
        A mean pooling 3-D layer with a output rank as 5.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 50, 50, 50, 32], name='input')
    >>> net = tl.layers.MeanPool3d(filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME')(net)
    >>> output shape : [None, 25, 25, 25, 32]

    """

    def __init__(
        self,
        filter_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding='VALID',
        data_format='channels_last',
        name=None  # 'meanpool3d'
    ):
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = self._strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MeanPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}' ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self._strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        outputs = tl.ops.avg_pool3d(
            input=inputs,
            ksize=self.filter_size,
            strides=self._strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs


class GlobalMaxPool1d(Module):
    """The :class:`GlobalMaxPool1d` class is a 1D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 30], name='input')
    >>> net = tl.layers.GlobalMaxPool1d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format="channels_last",
        name=None  # 'globalmaxpool1d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool1d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tl.ReduceMax(axis=1)
        elif self.data_format == 'channels_first':
            self.reduce_max = tl.ReduceMax(axis=2)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)
        return outputs


class GlobalMeanPool1d(Module):
    """The :class:`GlobalMeanPool1d` class is a 1D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 30], name='input')
    >>> net = tl.layers.GlobalMeanPool1d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmeanpool1d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMeanPool1d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_mean = tl.ReduceMean(axis=1)
        elif self.data_format == 'channels_first':
            self.reduce_mean = tl.ReduceMean(axis=2)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_mean(inputs)
        return outputs


class GlobalMaxPool2d(Module):
    """The :class:`GlobalMaxPool2d` class is a 2D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 100, 30], name='input')
    >>> net = tl.layers.GlobalMaxPool2d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmaxpool2d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool2d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tl.ReduceMax(axis=[1, 2])
        elif self.data_format == 'channels_first':
            self.reduce_max = tl.ReduceMax(axis=[2, 3])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)
        return outputs


class GlobalMeanPool2d(Module):
    """The :class:`GlobalMeanPool2d` class is a 2D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 100, 30], name='input')
    >>> net = tl.layers.GlobalMeanPool2d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmeanpool2d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMeanPool2d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_mean = tl.ReduceMean(axis=[1, 2])
        elif self.data_format == 'channels_first':
            self.reduce_mean = tl.ReduceMean(axis=[2, 3])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_mean(inputs)
        return outputs


class GlobalMaxPool3d(Module):
    """The :class:`GlobalMaxPool3d` class is a 3D Global Max Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 100, 100, 30], name='input')
    >>> net = tl.layers.GlobalMaxPool3d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmaxpool3d'
    ):
        super().__init__(name)

        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMaxPool3d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.reduce_max = tl.ReduceMax(axis=[1, 2, 3])
        elif self.data_format == 'channels_first':
            self.reduce_max = tl.ReduceMax(axis=[2, 3, 4])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )

    def forward(self, inputs):
        outputs = self.reduce_max(inputs)
        return outputs


class GlobalMeanPool3d(Module):
    """The :class:`GlobalMeanPool3d` class is a 3D Global Mean Pooling layer.

    Parameters
    ------------
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 100, 100, 100, 30], name='input')
    >>> net = tl.layers.GlobalMeanPool3d()(net)
    >>> output shape : [None, 30]

    """

    def __init__(
        self,
        data_format='channels_last',
        name=None  # 'globalmeanpool3d'
    ):
        super().__init__(name)
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("GlobalMeanPool3d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    def forward(self, inputs):
        if self.data_format == 'channels_last':
            outputs = tl.reduce_mean(input_tensor=inputs, axis=[1, 2, 3])
        elif self.data_format == 'channels_first':
            outputs = tl.reduce_mean(input_tensor=inputs, axis=[2, 3, 4])
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs


class CornerPool2d(Module):
    """Corner pooling for 2D image [batch, height, width, channel], see `here <https://arxiv.org/abs/1808.01244>`__.

    Parameters
    ----------
    mode : str
        TopLeft for the top left corner,
        Bottomright for the bottom right corner.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 32, 32, 8], name='input')
    >>> net = tl.layers.CornerPool2d(mode='TopLeft',name='cornerpool2d')(net)
    >>> output shape : [None, 32, 32, 8]

    """

    def __init__(
        self,
        mode='TopLeft',
        name=None  # 'cornerpool2d'
    ):
        super().__init__(name)
        self.mode = mode
        self.build()
        self._built = True

        logging.info("CornerPool2d %s : mode: %s" % (self.name, str(mode)))

    def __repr__(self):
        s = ('{classname}(mode={mode}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    def forward(self, inputs):
        _, input_width, input_height, _ = tl.get_tensor_shape(inputs)
        # input_width = inputs.shape[2]
        # input_height = inputs.shape[1]
        batch_min = tl.reduce_min(inputs)
        if self.mode == 'TopLeft':
            temp_bottom = tl.pad(
                inputs, tl.constant([[0, 0], [0, input_height - 1], [0, 0], [0, 0]]), constant_values=batch_min
            )
            temp_right = tl.pad(
                inputs, tl.constant([[0, 0], [0, 0], [0, input_width - 1], [0, 0]]), constant_values=batch_min
            )
            temp_bottom = tl.ops.max_pool(temp_bottom, ksize=(input_height, 1), strides=(1, 1), padding='VALID')
            temp_right = tl.ops.max_pool(temp_right, ksize=(1, input_width), strides=(1, 1), padding='VALID')
            outputs = tl.add(temp_bottom, temp_right)  #, name=self.name)
        elif self.mode == 'BottomRight':
            temp_top = tl.pad(
                inputs, tl.constant([[0, 0], [input_height - 1, 0], [0, 0], [0, 0]]), constant_values=batch_min
            )
            temp_left = tl.pad(
                inputs, tl.constant([[0, 0], [0, 0], [input_width - 1, 0], [0, 0]]), constant_values=batch_min
            )
            temp_top = tl.ops.max_pool(temp_top, ksize=(input_height, 1), strides=(1, 1), padding='VALID')
            temp_left = tl.ops.max_pool(temp_left, ksize=(1, input_width), strides=(1, 1), padding='VALID')
            outputs = tl.add(temp_top, temp_left)
        else:
            outputs = tl.identity(inputs)
        return outputs


class AdaptiveMeanPool1d(Module):
    """The :class:`AdaptiveMeanPool1d` class is a 1D Adaptive Mean Pooling layer.

    Parameters
    ------------
    output_size : int
        The target output size. It must be an integer.
    data_format : str
        One of channels_last (default, [batch,  width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None, 32, 3], name='input')
    >>> net = tl.layers.AdaptiveMeanPool1d(output_size=16)(net)
    >>> output shape : [None, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMeanPool1d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMeanPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")

        self.adaptivemeanpool1d = tl.ops.AdaptiveMeanPool1D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemeanpool1d(inputs)
        return outputs


class AdaptiveMeanPool2d(Module):
    """The :class:`AdaptiveMeanPool2d` class is a 2D Adaptive Mean Pooling layer.

    Parameters
    ------------
    output_size : int or list or  tuple
        The target output size. It cloud be an int \[int,int]\(int, int).
    data_format : str
        One of channels_last (default, [batch,  height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayer

    >>> net = tl.layers.Input([None,32, 32, 3], name='input')
    >>> net = tl.layers.AdaptiveMeanPool2d(output_size=16)(net)
    >>> output shape : [None,16, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMeanPool2d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMeanPool2d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
        else:
            raise Exception("unsupported data format")

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 2

        self.adaptivemeanpool2d = tl.ops.AdaptiveMeanPool2D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemeanpool2d(inputs)
        return outputs


class AdaptiveMeanPool3d(Module):
    """The :class:`AdaptiveMeanPool3d` class is a 3D Adaptive Mean Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int,int]\(int, int, int).
        data_format : str
            One of channels_last (default, [batch,  depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayer

        >>> net = tl.layers.Input([None,32, 32, 32, 3], name='input')
        >>> net = tl.layers.AdaptiveMeanPool3d(output_size=16)(net)
        >>> output shape : [None, 16, 16, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMeanPool3d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMeanPool3d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
        else:
            raise Exception("unsupported data format")

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 3

        self.adaptivemeanpool3d = tl.ops.AdaptiveMeanPool3D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemeanpool3d(inputs)
        return outputs


class AdaptiveMaxPool1d(Module):
    """The :class:`AdaptiveMaxPool1d` class is a 1D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int
            The target output size. It must be an integer.
        data_format : str
            One of channels_last (default, [batch,  width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayer

        >>> net = tl.layers.Input([None, 32, 3], name='input')
        >>> net = tl.layers.AdaptiveMaxPool1d(output_size=16)(net)
        >>> output shape : [None, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool1d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")

        self.adaptivemaxpool1d = tl.ops.AdaptiveMaxPool1D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemaxpool1d(inputs)
        return outputs


class AdaptiveMaxPool2d(Module):
    """The :class:`AdaptiveMaxPool2d` class is a 2D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int]\(int, int).
        data_format : str
            One of channels_last (default, [batch, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayer

        >>> net = tl.layers.Input([None, 32, 32, 3], name='input')
        >>> net = tl.layers.AdaptiveMaxPool2d(output_size=16)(net)
        >>> output shape : [None, 16, 16, 3]

    """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool2d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool1d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
        else:
            raise Exception("unsupported data format")
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 2

        self.adaptivemaxpool2d = tl.ops.AdaptiveMaxPool2D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemaxpool2d(inputs)
        return outputs


class AdaptiveMaxPool3d(Module):
    """The :class:`AdaptiveMaxPool3d` class is a 3D Adaptive Max Pooling layer.

        Parameters
        ------------
        output_size : int or list or  tuple
            The target output size. It cloud be an int \[int,int,int]\(int, int, int).
        data_format : str
            One of channels_last (default, [batch,  depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
        name : None or str
            A unique layer name.

        Examples
        ---------
        With TensorLayer

        >>> net = tl.layers.Input([None,32, 32, 32, 3], name='input')
        >>> net = tl.layers.AdaptiveMaxPool3d(output_size=16)(net)
        >>> output shape : [None, 16, 16, 16, 3]

        """

    def __init__(self, output_size, data_format='channels_last', name=None):
        super(AdaptiveMaxPool3d, self).__init__(name)
        self.output_size = output_size
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info("AdaptiveMaxPool3d %s: output_size: %s " % (self.name, str(output_size)))

    def __repr__(self):
        s = ('{classname}(output_size={output_size}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
        else:
            raise Exception("unsupported data format")

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, ) * 3

        self.adaptivemaxpool3d = tl.ops.AdaptiveMaxPool3D(output_size=self.output_size, data_format=self.data_format)

    def forward(self, inputs):

        outputs = self.adaptivemaxpool3d(inputs)
        return outputs
