#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Conv1dLayer',
    'Conv2dLayer',
    'Conv3dLayer',
]


class Conv1dLayer(Layer):
    """
    The :class:`Conv1dLayer` class is a 1D CNN layer, see `tf.nn.convolution <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_length, in_channels, out_channels).
    stride : int
        The number of entries by which the filter is moved right at a step.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : None or str
        A unique layer name

    """

    def __init__(
            self,
            act=None,
            shape=(5, 1, 5),
            stride=1,
            padding='SAME',
            data_format='NWC',
            dilation_rate=1,
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None,  #'cnn1d',
    ):
        # super(Conv1dLayer, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.act = act,
        self.shape = shape
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        logging.info(
            "Conv1dLayer %s: shape: %s stride: %s pad: %s act: %s" % (
                self.name, str(shape), str(stride), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs_shape):
        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\W_conv1d', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init, init_args=self.W_init_args)
        if self.b_init:
            self.b = self._get_weights(
                "biases",
                shape=(self.n_filter),  #self.shape[-1]),
                init=self.b_init,
                init_args=self.b_init_args
            )
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\b_conv1d', shape=(self.shape[-1]), initializer=self.b_init,
        #         dtype=LayersConfig.tf_dtype, **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):

        outputs = tf.nn.convolution(
            input=inputs,
            filters=self.W,
            strides=(self.stride, ),
            padding=self.padding,
            dilations=(self.dilation_rate, ),
            data_format=self.data_format,
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')

        outputs = self.act(outputs)
        return outputs


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    name : None or str
        A unique layer name.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = tl.layers.Input(x, name='input_layer')
    >>> net = tl.layers.Conv2dLayer(net,
    ...                   act = tf.nn.relu,
    ...                   shape = (5, 5, 1, 32),  # 32 features for each 5x5 patch
    ...                   strides = (1, 1, 1, 1),
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> net = tl.layers.Pool(net,
    ...                   ksize=(1, 2, 2, 1),
    ...                   strides=(1, 2, 2, 1),
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    Without TensorLayer, you can implement 2D convolution as follow.

    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )

    """

    def __init__(
            self,
            act=None,
            shape=(5, 5, 1, 100),
            strides=(1, 1, 1, 1),
            padding='SAME',
            data_format=None,
            dilations=[1, 1, 1, 1],
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            name=None,  #'cnn_layer',
    ):
        # super(Conv2dLayer, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.act = act,
        self.shape = shape
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        logging.info(
            "Conv2dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs):
        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init, init_args=self.W_init_args)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init, init_args=self.b_init_args)

        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\W_conv2d', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        # if self.b_init:
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\b_conv2d', shape=(self.shape[-1]), initializer=self.b_init,
        #         dtype=LayersConfig.tf_dtype, **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):
        outputs = tf.nn.conv2d(
            inputs,
            self.W,
            strides=self.strides,
            padding=self.padding,
            use_cudnn_on_gpu=self.use_cudnn_on_gpu,
            data_format=self.data_format,
            dilations=self.dilations,
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')

        if self.act:
            outputs = self.act(outputs)
        return outputs


class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
        Must be in the same order as the shape dimension.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        "NHWC" or "NCDHW", default is "NDHWC".
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : None or str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
    >>> n = tl.layers.Input(x, name='in3')
    >>> n = tl.layers.Conv3dLayer(n, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
    [None, 50, 50, 50, 32]
    """

    def __init__(
            self,
            act=None,
            shape=(2, 2, 2, 3, 32),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            data_format='NDHWC',
            dilations=[1, 1, 1, 1, 1],
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.02),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None,  #'cnn3d_layer',
    ):
        # super(Conv3dLayer, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.act = act,
        self.shape = shape
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        logging.info(
            "Conv3dLayer %s: shape: %s strides: %s pad: %s act: %s" % (
                self.name, str(shape), str(strides), padding,
                self.act.__name__ if self.act is not None else 'No Activation'
            )
        )

    def build(self, inputs):

        self.W = self._get_weights("filters", shape=self.shape, init=self.W_init, init_args=self.W_init_args)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_filter), init=self.b_init, init_args=self.b_init_args)

        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\W_conv3d', shape=self.shape, initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        #
        # if self.b_init:
        #     self.b = tf.compat.v1.get_variable(
        #         name=self.name + '\b_conv3d', shape=(self.shape[-1]), initializer=self.b_init,
        #         dtype=LayersConfig.tf_dtype, **self.b_init_args
        #     )
        #     self.add_weights([self.W, self.b])
        # else:
        #     self.add_weights(self.W)

    def forward(self, inputs):
        outputs = tf.nn.conv3d(
            input=inputs,
            filter=self.W,
            strides=self.strides,
            padding=self.padding,
            # use_cudnn_on_gpu=self.use_cudnn_on_gpu, #True,
            data_format=self.data_format,  #'NDHWC',
            dilations=self.dilation_rate,  #[1, 1, 1, 1, 1],
            name=self.name,
        )

        if self.b_init:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')

        if self.act:
            outputs = self.act(outputs)
        return outputs
