#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'Conv1d',
    'Conv2d',
]


class Conv1d(Layer):
    """Simplified version of :class:`Conv1dLayer`.

    Parameters
    ----------
    n_filter : int
        The number of filters
    filter_size : int
        The filter size
    stride : int
        The stride step
    dilation_rate : int
        Specifying the dilation rate to use for dilated convolution.
    act : activation function
        The function that is applied to the layer activations
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        channels_last 'NWC' (default) or channels_first.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (deprecated).
    b_init_args : dictionary
        The arguments for the bias vector initializer (deprecated).
    name : str
        A unique layer name

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (batch_size, width))
    >>> y_ = tf.placeholder(tf.int64, shape=(batch_size,))
    >>> n = InputLayer(x, name='in')
    >>> n = ReshapeLayer(n, (-1, width, 1), name='rs')
    >>> n = Conv1d(n, 64, 3, 1, act=tf.nn.relu, name='c1')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m1')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c2')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m2')
    >>> n = Conv1d(n, 128, 3, 1, act=tf.nn.relu, name='c3')
    >>> n = MaxPool1d(n, 2, 2, padding='valid', name='m3')
    >>> n = FlattenLayer(n, name='f')
    >>> n = DenseLayer(n, 500, tf.nn.relu, name='d1')
    >>> n = DenseLayer(n, 100, tf.nn.relu, name='d2')
    >>> n = DenseLayer(n, 2, None, name='o')
    """

    def __init__(
        self,
        n_filter=32,
        filter_size=5,
        stride=1,
        padding='valid',
        dilation_rate=1,
        data_format="channels_last",
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='conv1d'
    ):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("`data_format` value is not valid, should be either: 'channels_last' or 'channels_first'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(Conv1d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %d" % self.filter_size)
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % self.stride)
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("dilation_rate: %s" % self.dilation_rate)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        is_name_reuse = tf.get_variable_scope().reuse

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.conv1d(
                inputs=self._temp_data['inputs'],
                filters=self.n_filter,
                kernel_size=self.filter_size,
                strides=self.stride,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                activation=None,
                kernel_initializer=self.W_init,
                bias_initializer=self.b_init,
                use_bias=(True if self.b_init else False),
                reuse=is_name_reuse,
                trainable=self._temp_data['is_train'],
                name=None
            )

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


class Conv2d(Layer):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (for TF < 1.5).
    b_init_args : dictionary
        The arguments for the bias vector initializer (for TF < 1.5).
    data_format : str
        "channels_last" (NHWC, default) or "channels_first" (NCHW).
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A :class:`Conv2dLayer` object.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = InputLayer(x, name='inputs')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool2')
    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        dilation_rate=(1, 1),
        data_format="channels_last",
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='conv2d',
    ):

        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError("`data_format` value is not valid, should be either: 'channels_last' or 'channels_first'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(Conv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        try:
            additional_str.append("dilation_rate: %s" % str(self.dilation_rate))
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        is_name_reuse = tf.get_variable_scope().reuse

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.layers.conv2d(
                inputs=self._temp_data['inputs'],
                filters=self.n_filter,
                kernel_size=self.filter_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate,
                activation=None,
                kernel_initializer=self.W_init,
                bias_initializer=self.b_init,
                use_bias=(True if self.b_init else False),
                reuse=is_name_reuse,
                trainable=self._temp_data['is_train'],
                name=None
            )

        self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

        self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
