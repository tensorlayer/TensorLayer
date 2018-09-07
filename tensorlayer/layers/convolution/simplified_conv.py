#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Conv1d',
    'Conv2d',
]


class Conv1d(Layer):
    """Simplified version of :class:`Conv1dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=5, stride=1, dilation_rate=1, act=None, padding='SAME',
            data_format="channels_last", W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0), W_init_args=None, b_init_args=None, name='conv1d'
    ):
        super(Conv1d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "Conv1d %s: n_filter: %d filter_size: %s stride: %d pad: %s act: %s dilation_rate: %d" % (
                self.name, n_filter, filter_size, stride, padding,
                self.act.__name__ if self.act is not None else 'No Activation', dilation_rate
            )
        )

        _conv1d = tf.layers.Conv1D(
            filters=n_filter, kernel_size=filter_size, strides=stride, padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, activation=self.act, use_bias=(True if b_init else False),
            kernel_initializer=W_init, bias_initializer=b_init, name=name
        )

        # _conv1d.dtype = LayersConfig.tf_dtype   # unsupport, it will use the same dtype of inputs
        self.outputs = _conv1d(self.inputs)
        # new_variables = _conv1d.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)

        self._add_layers(self.outputs)
        self._add_params(new_variables)


class Conv2d(Layer):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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
    use_cudnn_on_gpu : bool
        Default is False (for TF < 1.5).
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=None,
            padding='SAME',
            data_format='channels_last',
            dilation_rate=(1, 1),
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            name='conv2d',
    ):
        # if len(strides) != 2:
        #     raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")

        # try:
        #     pre_channel = int(layer.outputs.get_shape()[-1])

        # except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        #     pre_channel = 1
        #     logging.info("[warnings] unknow input channels, set to 1")

        super(Conv2d, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "Conv2d %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )
        # with tf.variable_scope(name) as vs:
        conv2d = tf.layers.Conv2D(
            # inputs=self.inputs,
            filters=n_filter,
            kernel_size=filter_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=self.act,
            use_bias=(False if b_init is None else True),
            kernel_initializer=W_init,  # None,
            bias_initializer=b_init,  # f.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=name,
            # reuse=None,
        )
        self.outputs = conv2d(self.inputs)  # must put before ``new_variables``
        # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)  #vs.name)
        new_variables = get_collection_trainable(self.name)
        # new_variables = []
        # for p in tf.trainable_variables():
        #     # print(p.name.rpartition('/')[0], self.name)
        #     if p.name.rpartition('/')[0] == self.name:
        #         new_variables.append(p)
        # exit()
        # TF_GRAPHKEYS_VARIABLES  TF_GRAPHKEYS_VARIABLES
        # print(self.name, name)
        # print(tf.trainable_variables())#tf.GraphKeys.TRAINABLE_VARIABLES)
        # print(new_variables)
        # print(conv2d.weights)

        self._add_layers(self.outputs)
        self._add_params(new_variables)  # conv2d.weights)
