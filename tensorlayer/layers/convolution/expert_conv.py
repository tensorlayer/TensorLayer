#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

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
    shape : tuple of int
        The shape of the filters: (filter_length, in_channels, out_channels).
    stride : int
        The number of entries by which the filter is moved right at a step.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
    use_cudnn_on_gpu : bool
        Default is True.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name

    """

    def __init__(
        self,
        shape=(5, 1, 5),
        stride=1,
        padding='SAME',
        data_format='NWC',
        use_cudnn_on_gpu=True,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='conv1d_layer',
    ):

        if data_format not in ["NWC", "NCW"]:
            raise ValueError("`data_format` value is not valid, should be either: 'NWC' or 'NCW'")

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        self.shape = shape
        self.stride = stride
        self.padding = padding
        self.data_format = data_format
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(Conv1dLayer, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("shape: %s" % str(self.shape))
        except AttributeError:
            pass

        try:
            additional_str.append("stride: %s" % str(self.stride))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            weight_matrix = self._get_tf_variable(
                name='W_conv1d',
                shape=self.shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            self._temp_data['outputs'] = tf.nn.conv1d(
                self._temp_data['inputs'],
                weight_matrix,
                stride=self.stride,
                padding=self.padding,
                use_cudnn_on_gpu=self.use_cudnn_on_gpu,
                data_format=self.data_format
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_conv1d',
                    shape=(self.shape[-1], ),
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])


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
    use_cudnn_on_gpu : bool
        Default is True.
    gemmlowp_at_inference : boolean
        If True, use gemmlowp instead of ``tf.matmul`` (gemm) for inference. (TODO).
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer
    >>> import tensorflow as tf
    >>> import tensorlayer as tl

    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

    >>> net = tl.layers.InputLayer(x, name='input_layer')

    >>> net = tl.layers.Conv2dLayer(net,
    ...                   act = tf.nn.relu,
    ...                   shape = (5, 5, 1, 32),  # 32 features for each 5x5 patch
    ...                   strides = (1, 1, 1, 1),
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   name ='conv2d_1')     # output: (?, 28, 28, 32)

    >>> net = tl.layers.PoolLayer(net,
    ...                   ksize=(1, 2, 2, 1),
    ...                   strides=(1, 2, 2, 1),
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1')   # output: (?, 14, 14, 32)
    """

    def __init__(
        self,
        shape=(5, 5, 1, 100),
        strides=(1, 1, 1, 1),
        padding='SAME',
        data_format="NHWC",
        use_cudnn_on_gpu=True,
        gemmlowp_at_inference=False,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='conv2d_layer'
    ):

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        if len(strides) != 4:
            raise ValueError("len(strides) should be 4.")

        if data_format not in ["NHWC", "NCHW"]:
            raise ValueError("`data_format` value is not valid, should be either: 'NHWC' or 'NCHW'")

        # TODO: Implement GEMM
        if gemmlowp_at_inference:
            raise NotImplementedError("TODO. The current version use tf.matmul for inferencing.")

        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.gemmlowp_at_inference = gemmlowp_at_inference
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(Conv2dLayer, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("shape: %s" % str(self.shape))
        except AttributeError:
            pass

        try:
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            weight_matrix = self._get_tf_variable(
                name='W_conv2d',
                shape=self.shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            self._temp_data['outputs'] = tf.nn.conv2d(
                self._temp_data['inputs'],
                weight_matrix,
                strides=self.strides,
                padding=self.padding,
                use_cudnn_on_gpu=self.use_cudnn_on_gpu,
                data_format=self.data_format
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_conv2d',
                    shape=(self.shape[-1]),
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])


class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`__.

    Parameters
    ----------
    shape : tuple of int
        Shape of the filters: (filter_depth, filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
        Must be in the same order as the shape dimension.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
    >>> n = tl.layers.InputLayer(x, name='in3')
    >>> n = tl.layers.Conv3dLayer(n, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
    [None, 50, 50, 50, 32]
    """

    def __init__(
        self,
        shape=(2, 2, 2, 3, 32),
        strides=(1, 2, 2, 2, 1),
        padding='SAME',
        data_format='NDHWC',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='conv3d_layer'
    ):

        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError("`padding` value is not valid, should be either: 'SAME' or 'VALID'")

        if data_format not in ["NDHWC", "NCDHW"]:
            raise ValueError("`data_format` value is not valid, should be either: 'NDHWC' or 'NCDHW'")

        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(Conv3dLayer, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("shape: %s" % str(self.shape))
        except AttributeError:
            pass

        try:
            additional_str.append("strides: %s" % str(self.strides))
        except AttributeError:
            pass

        try:
            additional_str.append("padding: %s" % self.padding)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):
            weight_matrix = self._get_tf_variable(
                name='W_conv3d',
                shape=self.shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            self._temp_data['outputs'] = tf.nn.conv3d(
                self._temp_data['inputs'],
                weight_matrix,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format
            )

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_conv3d',
                    shape=(self.shape[-1]),
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                self._temp_data['outputs'] = tf.nn.bias_add(self._temp_data['outputs'], b, name='bias_add')

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
