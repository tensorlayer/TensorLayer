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
    filter_size : tuple of int
        The size of the window for each dimension of the input tensor.
        Note that: len(filter_size) >= 4.
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
            filter_size=(1, 2, 2, 1),
            strides=(1, 2, 2, 1),
            padding='SAME',
            pool=tf.nn.max_pool,
            name=None,  #'pool_pro',
    ):
        # super(PoolLayer, self).__init__(prev_layer=prev_layer, name=name)
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
        pass

    def forward(self, inputs):
        outputs = self.pool(
            inputs,
            ksize=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            name=self.name
        )
        return outputs


class MaxPool1d(Layer):
    """Max pooling for 1D signal.

    Parameters
    ----------
    filter_size : tuple of int
        Pooling window size.
    strides : int
        Stride of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, length, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            filter_size=3,
            strides=2,
            padding='SAME',
            data_format='channels_last',
            name=None,  #'maxpool1d'
    ):
        # super(MaxPool1d, self).__init__(prev_layer=prev_layer, name=name)
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
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # https://www.tensorflow.org/api_docs/python/tf/nn/pool
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")
        self._filter_size = [self.filter_size]
        self._strides = [self.strides]

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`  # NO PREVIOUS LAYER NOW
            The previous layer with a output rank as 3 [batch, length(width), channel].
        """
        ## TODO : tf.layers will be removed in TF 2.0
        # outputs = tf.layers.max_pooling1d(
        #     inputs, self.filter_size, self.strides, padding=self.padding, data_format=self.data_format, name=self.name
        # )
        # https://www.tensorflow.org/api_docs/python/tf/nn/pool
        # print(self.strides, self.data_format)
        outputs = tf.nn.pool(
            input=inputs,
            window_shape=self._filter_size,
            pooling_type="MAX",
            padding=self.padding,
            dilations=None, # TODO: support dilations
            strides=self._strides,
            name=self.name,
            data_format=self.data_format
        )
        return outputs


# x = tf.placeholder("float32", [None, 100, 3])
# n = MaxPool1d(name='sasds')
# print(type(n))
# n.build(x)
# print(n.strides, n.filter_size, n.data_format)
# # exit()
# y = n.forward(x)
# print(type(y), y)#.outputs)
# exit()


class MeanPool1d(Layer):
    """Mean pooling for 1D signal.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 3.
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

    # logging.info("MeanPool1d %s: filter_size: %s strides: %s padding: %s" % (self.name, str(filter_size), str(strides), str(padding)))
    # outputs = tf.layers.average_pooling1d(prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name)
    #
    # net_new = copy.copy(prev_layer)
    # net_new.outputs = outputs
    # net_new.all_layers.extend([outputs])
    # return net_new
    def __init__(
            self,  #prev_layer,
            filter_size=3,
            strides=2,
            padding='SAME',
            data_format='channels_last',
            name=None,  #'meanpool1d'
    ):
        # super(MeanPool1d, self).__init__(prev_layer=prev_layer, name=name)
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
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        # pass
        # https://www.tensorflow.org/api_docs/python/tf/nn/pool
        if self.data_format == 'channels_last':
            self.data_format = 'NWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCW'
        else:
            raise Exception("unsupported data format")
        self._filter_size = [self.filter_size]
        self._strides = [self.strides]

    def forward(self, inputs):
        # self.outputs = tf.layers.average_pooling1d(
        #     prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        # self._add_layers(self.outputs)
        # https://www.tensorflow.org/api_docs/python/tf/nn/pool
        outputs = tf.nn.pool(
            input=inputs,
            window_shape=self._filter_size,
            pooling_type="AVG",
            padding=self.padding,
            dilations=None, # TODO: support dilations
            strides=self._strides,
            name=self.name,
            data_format=self.data_format
        )
        return outputs


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
            self,
            filter_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            data_format='channels_last',
            name=None,  #'maxpool2d'
    ):
        if strides is None:
            strides = filter_size

        # super(MaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.strides = [1, self.strides[0], self.strides[1], 1]
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4.
        """
        # outputs = tf.layers.max_pooling2d(
        #     inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        outputs = tf.nn.max_pool(
            input=inputs,
            ksize=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            name=self.name
        )
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
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 4 [batch, height, width, channel].
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
            self,  #prev_layer,
            filter_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            data_format='channels_last',
            name=None,  #'meanpool2d'
    ):
        if strides is None:
            strides = filter_size

        # super(MeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MeanPool2d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.strides = [1, self.strides[0], self.strides[1], 1]
        if self.data_format == 'channels_last':
            self.data_format = 'NHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCHW'
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4.
        """
        # self.outputs = tf.layers.average_pooling2d(
        #     self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        # self._add_layers(self.outputs)
        outputs = tf.nn.avg_pool(
            input=inputs,
            ksize=self.filter_size,
            strides=self.strides,
            padding=self.padding,
            name=self.name
        )
        return outputs


class MaxPool3d(Layer):
    """Max pooling for 3D volume.

    Parameters
    ------------
    # prev_layer : :class:`Layer`
    #     The previous layer with a output rank as 5.
    filter_size : tuple of int
        Pooling window size.
    strides : tuple of int
        Strides of the pooling operation.
    padding : str
        The padding method: 'valid' or 'same'.
    data_format : str
        One of channels_last (default, [batch, depth, height, width, channel]) or channels_first. The ordering of the dimensions in the inputs.
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A max pooling 3-D layer with a output rank as 5.

    """

    def __init__(
            self,  #prev_layer,
            filter_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='valid',
            data_format='channels_last',
            name=None,  #'maxpool3d'
    ):
        # super(MaxPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MaxPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5.
        """
        # self.outputs = tf.layers.max_pooling3d(
        #     self.inputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        # self._add_layers(self.outputs)
        outputs = tf.nn.max_pool3d(
            input=inputs,
            ksize=self.filter_size,
            strides=self.strides,
            padding=self.filter_size,
            data_format=self.data_format,
            name=self.name,
        )
        return outputs


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
    name : None or str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A mean pooling 3-D layer with a output rank as 5.

    """

    def __init__(
            self,  #prev_layer,
            filter_size=(3, 3, 3),
            strides=(2, 2, 2),
            padding='valid',
            data_format='channels_last',
            name=None,  #'meanpool3d'
    ):

        # super(MeanPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.build()
        self._built = True

        logging.info(
            "MeanPool3d %s: filter_size: %s strides: %s padding: %s" %
            (self.name, str(filter_size), str(strides), str(padding))
        )

    def __repr__(self):
        s = ('{classname}(filter_size={filter_size}'
             ', strides={strides}, padding={padding}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.strides = [1, self.strides[0], self.strides[1], self.strides[2], 1]
        if self.data_format == 'channels_last':
            self.data_format = 'NDHWC'
        elif self.data_format == 'channels_first':
            self.data_format = 'NCDHW'
        else:
            raise Exception("unsupported data format")

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5.
        """
        # self.outputs = tf.layers.average_pooling3d(
        #     prev_layer.outputs, filter_size, strides, padding=padding, data_format=data_format, name=name
        # )
        # self._add_layers(self.outputs)
        outputs = tf.nn.avg_pool3d(
            input=inputs,
            ksize=self.filter_size,
            strides=self.strides,
            padding=self.filter_size,
            data_format=self.data_format,
            name=self.name,
        )
        return outputs


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

    def __init__(self, data_format="channels_last", name=None):  #'globalmaxpool1d'):
        # super(GlobalMaxPool1d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)

        self.data_format= data_format
        self.name = name

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
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(input_tensor=inputs, axis=1, name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_max(input_tensor=inputs, axis=2, name=self.name)
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

    def __init__(self, data_format='channels_last', name=None):  #'globalmeanpool1d'):
        # super(GlobalMeanPool1d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        self.name = name

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
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 3 [batch, length, channel] or [batch, channel, length].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=1, name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=2, name=self.name)
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

    def __init__(self, data_format='channels_last', name=None):  #'globalmaxpool2d'):
        # super(GlobalMaxPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        self.name = name

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
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(input_tensor=inputs, axis=[1, 2], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_max(input_tensor=inputs, axis=[2, 3], name=self.name)
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

    def __init__(self, data_format='channels_last', name=None):  #'globalmeanpool2d'):
        # super(GlobalMeanPool2d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)

        self.data_format = data_format
        self.name = name

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
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 4 [batch, height, width, channel] or [batch, channel, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=[2, 3], name=self.name)
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

    def __init__(self, data_format='channels_last', name=None):  #'globalmaxpool3d'):
        # super(GlobalMaxPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)

        self.data_format = data_format
        self.name = name

        self.build()
        self._built = True

        logging.info("GlobalMaxPool3d %s" % self.name)

    def __repr__(self):
        s = '{classname}('
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_max(input_tensor=inputs, axis=[1, 2, 3], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_max(input_tensor=inputs, axis=[2, 3, 4], name=self.name)
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

    def __init__(self, data_format='channels_last', name=None):  #'globalmeanpool3d'):
        # super(GlobalMeanPool3d, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.data_format = data_format
        self.name = name

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
        """
        prev_layer : :class:`Layer`
            The previous layer with a output rank as 5 [batch, depth, height, width, channel] or [batch, channel, depth, height, width].
        """
        if self.data_format == 'channels_last':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=[1, 2, 3], name=self.name)
        elif self.data_format == 'channels_first':
            outputs = tf.reduce_mean(input_tensor=inputs, axis=[2, 3, 4], name=self.name)
        else:
            raise ValueError(
                "`data_format` should have one of the following values: [`channels_last`, `channels_first`]"
            )
        return outputs
