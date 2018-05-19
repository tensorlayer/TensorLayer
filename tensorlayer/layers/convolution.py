# -*- coding: utf-8 -*-

import tensorflow as tf

from .. import _logging as logging
from .core import *

from ..deprecation import deprecated_alias

__all__ = [
    'Conv1dLayer',
    'Conv2dLayer',
    'DeConv2dLayer',
    'Conv3dLayer',
    'DeConv3dLayer',
    'UpSampling2dLayer',
    'DownSampling2dLayer',
    'DeformableConv2d',
    'AtrousConv1dLayer',
    'AtrousConv2dLayer',
    'deconv2d_bilinear_upsampling_initializer',
    'Conv1d',
    'Conv2d',
    'DeConv2d',
    'DeConv3d',
    'DepthwiseConv2d',
    'SeparableConv1d',
    'SeparableConv2d',
    'GroupConv2d',
]


class Conv1dLayer(Layer):
    """
    The :class:`Conv1dLayer` class is a 1D CNN layer, see `tf.nn.convolution <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_length, in_channels, out_channels).
    stride : int
        The number of entries by which the filter is moved right at a step.
    dilation_rate : int
        Filter up-sampling/input down-sampling rate.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            shape=(5, 1, 5),
            stride=1,
            padding='SAME',
            data_format=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='cnn1d',
    ):
        super(Conv1dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "Conv1dLayer %s: shape:%s stride:%s pad:%s act:%s" % (name, str(shape), str(stride), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if act is None:
            act = tf.identity

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        if data_format not in [None, 'NWC', 'channels_last', 'NCW', 'channels_first']:
            raise ValueError("`data_format` should be among 'NWC', 'channels_last', 'NCW', 'channels_first'")

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_conv1d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            self.outputs = tf.nn.conv1d(self.inputs, W, stride=stride, padding=padding)  # 1.2
            if b_init:
                b = tf.get_variable(
                    name='b_conv1d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = self.outputs + b

            self.outputs = act(self.outputs)

        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (filter_height, filter_width, in_channels, out_channels).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
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
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    With TensorLayer

    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.Conv2dLayer(net,
    ...                   act = tf.nn.relu,
    ...                   shape = (5, 5, 1, 32),  # 32 features for each 5x5 patch
    ...                   strides = (1, 1, 1, 1),
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> net = tl.layers.PoolLayer(net,
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            shape=(5, 5, 1, 100),
            strides=(1, 1, 1, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            name='cnn_layer',
    ):
        super(Conv2dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "Conv2dLayer %s: shape:%s strides:%s pad:%s act:%s" %
            (name, str(shape), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        # dic = {'channels_last': 'NHWC', 'channels_first': 'NCHW'}
        if data_format not in [None, 'NHWC', 'channels_last', 'NCHW', 'channels_first']:
            raise ValueError("'data_format' must be among 'NHWC', 'channels_last', 'NCHW', 'channels_first'.")

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            if b_init:
                b = tf.get_variable(
                    name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.conv2d(
                        self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format
                    ) + b
                )
            else:
                self.outputs = act(
                    tf.nn.conv2d(
                        self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format
                    )
                )

        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class DeConv2dLayer(Layer):
    """A de-convolution 2D layer.

    See `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        Shape of the filters: (height, width, output_channels, in_channels).
        The filter's ``in_channels`` dimension must match that of value.
    output_shape : tuple of int
        Output shape of the deconvolution,
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for initializing the weight matrix.
    b_init_args : dictionary
        The arguments for initializing the bias vector.
    name : str
        A unique layer name.

    Notes
    -----
    - We recommend to use `DeConv2d` with TensorFlow version higher than 1.3.
    - shape = [h, w, the number of output channels of this layer, the number of output channel of the previous layer].
    - output_shape = [batch_size, any, any, the number of output channels of this layer].
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    A part of the generator in DCGAN example

    >>> batch_size = 64
    >>> inputs = tf.placeholder(tf.float32, [batch_size, 100], name='z_noise')
    >>> net_in = tl.layers.InputLayer(inputs, name='g/in')
    >>> net_h0 = tl.layers.DenseLayer(net_in, n_units = 8192,
    ...                            W_init = tf.random_normal_initializer(stddev=0.02),
    ...                            act = tf.identity, name='g/h0/lin')
    >>> print(net_h0.outputs._shape)
    ... (64, 8192)
    >>> net_h0 = tl.layers.ReshapeLayer(net_h0, shape=(-1, 4, 4, 512), name='g/h0/reshape')
    >>> net_h0 = tl.layers.BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train, name='g/h0/batch_norm')
    >>> print(net_h0.outputs._shape)
    ... (64, 4, 4, 512)
    >>> net_h1 = tl.layers.DeConv2dLayer(net_h0,
    ...                            shape=(5, 5, 256, 512),
    ...                            output_shape=(batch_size, 8, 8, 256),
    ...                            strides=(1, 2, 2, 1),
    ...                            act=tf.identity, name='g/h1/decon2d')
    >>> net_h1 = tl.layers.BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train, name='g/h1/batch_norm')
    >>> print(net_h1.outputs._shape)
    ... (64, 8, 8, 256)

    U-Net

    >>> ....
    >>> conv10 = tl.layers.Conv2dLayer(conv9, act=tf.nn.relu,
    ...        shape=(3,3,1024,1024), strides=(1,1,1,1), padding='SAME',
    ...        W_init=w_init, b_init=b_init, name='conv10')
    >>> print(conv10.outputs)
    ... (batch_size, 32, 32, 1024)
    >>> deconv1 = tl.layers.DeConv2dLayer(conv10, act=tf.nn.relu,
    ...         shape=(3,3,512,1024), strides=(1,2,2,1), output_shape=(batch_size,64,64,512),
    ...         padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            shape=(3, 3, 128, 256),
            output_shape=(1, 256, 256, 128),
            strides=(1, 2, 2, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='decnn2d_layer',
    ):
        super(DeConv2dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DeConv2dLayer %s: shape:%s out_shape:%s strides:%s pad:%s act:%s" %
            (name, str(shape), str(output_shape), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        # logging.info("  DeConv2dLayer: Untested")
        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_deconv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            if b_init:
                b = tf.get_variable(
                    name='b_deconv2d', shape=(shape[-2]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding)
                    + b
                )
            else:
                self.outputs = act(
                    tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding)
                )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
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
    name : str
        A unique layer name.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
    >>> n = tl.layers.InputLayer(x, name='in3')
    >>> n = tl.layers.Conv3dLayer(n, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
    ... [None, 50, 50, 50, 32]
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            shape=(2, 2, 2, 3, 32),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            data_format=None,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='cnn3d_layer',
    ):
        super(Conv3dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "Conv3dLayer %s: shape:%s strides:%s pad:%s act:%s" %
            (name, str(shape), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        if data_format not in [None, 'NDHWC', 'channels_last', 'NCDHW', 'channels_first']:
            raise ValueError("'data_format' must be one of 'channels_last', 'channels_first'.")

        with tf.variable_scope(name):
            # W = tf.Variable(W_init(shape=shape, **W_init_args), name='W_conv')
            # b = tf.Variable(b_init(shape=[shape[-1]], **b_init_args), name='b_conv')
            W = tf.get_variable(
                name='W_conv3d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            if b_init:
                b = tf.get_variable(
                    name='b_conv3d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, data_format=data_format, name=None) +
                    b
                )
            else:
                self.outputs = act(
                    tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, data_format=data_format, name=None)
                )

        # self.outputs = act( tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, name=None) + b )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class DeConv3dLayer(Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    act : activation function
        The activation function of this layer.
    shape : tuple of int
        The shape of the filters: (depth, height, width, output_channels, in_channels).
        The filter's in_channels dimension must match that of value.
    output_shape : tuple of int
        The output shape of the deconvolution.
    strides : tuple of int
        The sliding window strides for corresponding input dimensions.
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
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            shape=(2, 2, 2, 128, 256),
            output_shape=(1, 12, 32, 32, 128),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='decnn3d_layer',
    ):
        super(DeConv3dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DeConv3dLayer %s: shape:%s out_shape:%s strides:%s pad:%s act:%s" %
            (name, str(shape), str(output_shape), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_deconv3d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            if b_init:
                b = tf.get_variable(
                    name='b_deconv3d', shape=(shape[-2]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding)
                    + b
                )
            else:
                self.outputs = act(
                    tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding)
                )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class UpSampling2dLayer(Layer):
    """The :class:`UpSampling2dLayer` class is a up-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with 4-D Tensor of the shape (batch, height, width, channels) or 3-D Tensor of the shape (height, width, channels).
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is a scale factor; otherwise, the `size` is the numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, align the corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='upsample2d_layer',
    ):
        super(UpSampling2dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "UpSampling2dLayer %s: is_scale:%s size:%s method:%d align_corners:%s" %
            (name, is_scale, size, method, align_corners)
        )

        self.inputs = prev_layer.outputs

        if not isinstance(size, (list, tuple)) and len(size) == 2:
            raise AssertionError()

        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [int(size_h), int(size_w)]

        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [int(size_h), int(size_w)]

        else:
            raise Exception("Donot support shape %s" % self.inputs.get_shape())

        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(
                    self.inputs, size=size, method=method, align_corners=align_corners
                )
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(
                    self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners
                )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)


class DownSampling2dLayer(Layer):
    """The :class:`DownSampling2dLayer` class is down-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is the scale factor; otherwise, the `size` are numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, exactly align all 4 corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='downsample2d_layer',
    ):
        super(DownSampling2dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DownSampling2dLayer %s: is_scale:%s size:%s method:%d, align_corners:%s" %
            (name, is_scale, size, method, align_corners)
        )

        self.inputs = prev_layer.outputs

        if not isinstance(size, (list, tuple)) and len(size) == 2:
            raise AssertionError()

        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [int(size_h), int(size_w)]
        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [int(size_h), int(size_w)]
        else:
            raise Exception("Do not support shape %s" % self.inputs.get_shape())

        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(
                    self.inputs, size=size, method=method, align_corners=align_corners
                )
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(
                    self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners
                )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)


class DeformableConv2d(Layer):
    """The :class:`DeformableConv2d` class is a 2D
    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    offset_layer : :class:`Layer`
        To predict the offset of convolution operations.
        The output shape is (batchsize, input height, input width, 2*(number of element in the convolution kernel))
        e.g. if apply a 3*3 kernel, the number of the last dimension should be 18 (2*3*3)
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    act : activation function
        The activation function of this layer.
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

    Examples
    --------
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> offset1 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=act, padding='SAME', name='offset1')
    >>> net = tl.layers.DeformableConv2d(net, offset1, 32, (3, 3), act=act, name='deformable1')
    >>> offset2 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=act, padding='SAME', name='offset2')
    >>> net = tl.layers.DeformableConv2d(net, offset2, 64, (3, 3), act=act, name='deformable2')

    References
    ----------
    - The deformation operation was adapted from the implementation in `here <https://github.com/felixlaumon/deform-conv>`__

    Notes
    -----
    - The padding is fixed to 'SAME'.
    - The current implementation is not optimized for memory usgae. Please use it carefully.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            offset_layer=None,
            # shape=(3, 3, 1, 100),
            n_filter=32,
            filter_size=(3, 3),
            act=tf.identity,
            name='deformable_conv_2d',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None
    ):

        if tf.__version__ < "1.4":
            raise Exception(
                "Deformable CNN layer requires tensrflow 1.4 or higher version | current version %s" % tf.__version__
            )

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        def _to_bc_h_w(x, x_shape):
            """(b, h, w, c) -> (b*c, h, w)"""
            x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
            return x

        def _to_b_h_w_n_c(x, x_shape):
            """(b*c, h, w, n) -> (b, h, w, n, c)"""
            x = tf.reshape(x, (-1, x_shape[4], x_shape[1], x_shape[2], x_shape[3]))
            x = tf.transpose(x, [0, 2, 3, 4, 1])
            return x

        def tf_flatten(a):
            """Flatten tensor"""
            return tf.reshape(a, [-1])

        def _get_vals_by_coords(inputs, coords, idx, out_shape):
            indices = tf.stack([idx, tf_flatten(coords[:, :, :, :, 0]), tf_flatten(coords[:, :, :, :, 1])], axis=-1)
            vals = tf.gather_nd(inputs, indices)
            vals = tf.reshape(vals, out_shape)
            return vals

        def _tf_repeat(a, repeats):
            """Tensorflow version of np.repeat for 1D"""
            # https://github.com/tensorflow/tensorflow/issues/8521
            assert len(a.get_shape()) == 1

            a = tf.expand_dims(a, -1)
            a = tf.tile(a, [1, repeats])
            a = tf_flatten(a)
            return a

        def _tf_batch_map_coordinates(inputs, coords):
            """Batch version of tf_map_coordinates

            Only supports 2D feature maps

            Parameters
            ----------
            inputs : ``tf.Tensor``
                shape = (b*c, h, w)
            coords : ``tf.Tensor``
                shape = (b*c, h, w, n, 2)

            Returns
            -------
            ``tf.Tensor``
                A Tensor with the shape as (b*c, h, w, n)

            """
            input_shape = inputs.get_shape()
            coords_shape = coords.get_shape()
            batch_channel = tf.shape(inputs)[0]
            input_h = int(input_shape[1])
            input_w = int(input_shape[2])
            kernel_n = int(coords_shape[3])
            n_coords = input_h * input_w * kernel_n

            coords_lt = tf.cast(tf.floor(coords), 'int32')
            coords_rb = tf.cast(tf.ceil(coords), 'int32')
            coords_lb = tf.stack([coords_lt[:, :, :, :, 0], coords_rb[:, :, :, :, 1]], axis=-1)
            coords_rt = tf.stack([coords_rb[:, :, :, :, 0], coords_lt[:, :, :, :, 1]], axis=-1)

            idx = _tf_repeat(tf.range(batch_channel), n_coords)

            vals_lt = _get_vals_by_coords(inputs, coords_lt, idx, (batch_channel, input_h, input_w, kernel_n))
            vals_rb = _get_vals_by_coords(inputs, coords_rb, idx, (batch_channel, input_h, input_w, kernel_n))
            vals_lb = _get_vals_by_coords(inputs, coords_lb, idx, (batch_channel, input_h, input_w, kernel_n))
            vals_rt = _get_vals_by_coords(inputs, coords_rt, idx, (batch_channel, input_h, input_w, kernel_n))

            coords_offset_lt = coords - tf.cast(coords_lt, 'float32')

            vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, :, :, 0]
            vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, :, :, 0]
            mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, :, :, 1]

            return mapped_vals

        def _tf_batch_map_offsets(inputs, offsets, grid_offset):
            """Batch map offsets into input

            Parameters
            ------------
            inputs : ``tf.Tensor``
                shape = (b, h, w, c)
            offsets: ``tf.Tensor``
                shape = (b, h, w, 2*n)
            grid_offset: `tf.Tensor``
                Offset grids shape = (h, w, n, 2)

            Returns
            -------
            ``tf.Tensor``
                A Tensor with the shape as (b, h, w, c)

            """
            input_shape = inputs.get_shape()
            batch_size = tf.shape(inputs)[0]
            kernel_n = int(int(offsets.get_shape()[3]) / 2)
            input_h = input_shape[1]
            input_w = input_shape[2]
            channel = input_shape[3]

            # inputs (b, h, w, c) --> (b*c, h, w)
            inputs = _to_bc_h_w(inputs, input_shape)

            # offsets (b, h, w, 2*n) --> (b, h, w, n, 2)
            offsets = tf.reshape(offsets, (batch_size, input_h, input_w, kernel_n, 2))
            # offsets (b, h, w, n, 2) --> (b*c, h, w, n, 2)
            # offsets = tf.tile(offsets, [channel, 1, 1, 1, 1])

            coords = tf.expand_dims(grid_offset, 0)  # grid_offset --> (1, h, w, n, 2)
            coords = tf.tile(coords, [batch_size, 1, 1, 1, 1]) + offsets  # grid_offset --> (b, h, w, n, 2)

            # clip out of bound
            coords = tf.stack(
                [
                    tf.clip_by_value(coords[:, :, :, :, 0], 0.0, tf.cast(input_h - 1, 'float32')),
                    tf.clip_by_value(coords[:, :, :, :, 1], 0.0, tf.cast(input_w - 1, 'float32'))
                ], axis=-1
            )
            coords = tf.tile(coords, [channel, 1, 1, 1, 1])

            mapped_vals = _tf_batch_map_coordinates(inputs, coords)
            # (b*c, h, w, n) --> (b, h, w, n, c)
            mapped_vals = _to_b_h_w_n_c(mapped_vals, [batch_size, input_h, input_w, kernel_n, channel])

            return mapped_vals

        super(DeformableConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DeformableConv2d %s: n_filter: %d, filter_size: %s act:%s" %
            (name, n_filter, str(filter_size), act.__name__)
        )

        self.inputs = prev_layer.outputs

        self.offset_layer = offset_layer
        if act is None:
            act = tf.identity

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")
        shape = (filter_size[0], filter_size[1], pre_channel, n_filter)

        with tf.variable_scope(name):
            offset = self.offset_layer.outputs
            assert offset.get_shape()[-1] == 2 * shape[0] * shape[1]

            # Grid initialisation
            input_h = int(self.inputs.get_shape()[1])
            input_w = int(self.inputs.get_shape()[2])
            kernel_n = shape[0] * shape[1]
            initial_offsets = tf.stack(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]),
                                                   indexing='ij'))  # initial_offsets --> (kh, kw, 2)
            initial_offsets = tf.reshape(initial_offsets, (-1, 2))  # initial_offsets --> (n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, 1, n, 2)
            initial_offsets = tf.tile(initial_offsets, [input_h, input_w, 1, 1])  # initial_offsets --> (h, w, n, 2)
            initial_offsets = tf.cast(initial_offsets, 'float32')
            grid = tf.meshgrid(
                tf.range(-int((shape[0] - 1) / 2.0), int(input_h - int((shape[0] - 1) / 2.0)), 1),
                tf.range(-int((shape[1] - 1) / 2.0), int(input_w - int((shape[1] - 1) / 2.0)), 1), indexing='ij'
            )

            grid = tf.stack(grid, axis=-1)
            grid = tf.cast(grid, 'float32')  # grid --> (h, w, 2)
            grid = tf.expand_dims(grid, 2)  # grid --> (h, w, 1, 2)
            grid = tf.tile(grid, [1, 1, kernel_n, 1])  # grid --> (h, w, n, 2)
            grid_offset = grid + initial_offsets  # grid_offset --> (h, w, n, 2)

            input_deform = _tf_batch_map_offsets(self.inputs, offset, grid_offset)

            W = tf.get_variable(
                name='W_deformableconv2d', shape=[1, 1, shape[0] * shape[1], shape[-2], shape[-1]], initializer=W_init,
                dtype=LayersConfig.tf_dtype, **W_init_args
            )

            if b_init:
                b = tf.get_variable(
                    name='b_deformableconv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype,
                    **b_init_args
                )

                self.outputs = tf.reshape(
                    tensor=act(tf.nn.conv3d(input_deform, W, strides=[1, 1, 1, 1, 1], padding='VALID', name=None) + b),
                    shape=(tf.shape(self.inputs)[0], input_h, input_w, shape[-1])
                )
            else:
                self.outputs = tf.reshape(
                    tensor=act(tf.nn.conv3d(input_deform, W, strides=[1, 1, 1, 1, 1], padding='VALID', name=None)),
                    shape=[tf.shape(self.inputs)[0], input_h, input_w, shape[-1]]
                )

        # fixed
        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)

        # add offset_layer properties
        # offset_params = [osparam for osparam in offset_layer.all_params if osparam not in layer.all_params]
        # offset_layers = [oslayer for oslayer in offset_layer.all_layers if oslayer not in layer.all_layers]
        #
        # self.all_params.extend(list(offset_params))
        # self.all_layers.extend(list(offset_layers))
        # self.all_drop.update(dict(offset_layer.all_drop))

        # this layer
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


@deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
def atrous_conv1d(
        prev_layer,
        n_filter=32,
        filter_size=2,
        stride=1,
        dilation=1,
        act=tf.identity,
        padding='SAME',
        data_format='NWC',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        name='conv1d',
):
    """Simplified version of :class:`AtrousConv1dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : int
        The filter size.
    stride : tuple of int
        The strides: (height, width).
    dilation : int
        The filter dilation size.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    data_format : str
        Default is 'NWC' as it is a 1D CNN.
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

    Returns
    -------
    :class:`Layer`
        A :class:`AtrousConv1dLayer` object

    """

    if W_init_args is None:
        W_init_args = {}
    if b_init_args is None:
        b_init_args = {}

    return Conv1dLayer(
        prev_layer=prev_layer,
        act=act,
        shape=(filter_size, int(prev_layer.outputs.get_shape()[-1]), n_filter),
        stride=stride,
        padding=padding,
        dilation_rate=dilation,
        data_format=data_format,
        W_init=W_init,
        b_init=b_init,
        W_init_args=W_init_args,
        b_init_args=b_init_args,
        name=name,
    )


class AtrousConv2dLayer(Layer):
    """The :class:`AtrousConv2dLayer` class is 2D atrous convolution (a.k.a. convolution with holes or dilated
    convolution) 2D layer, see `tf.nn.atrous_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#atrous_conv2d>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with a 4D output tensor in the shape of (batch, height, width, channels).
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size: (height, width).
    rate : int
        The stride that we sample input values in the height and width dimensions.
        This equals the rate that we up-sample the filters by inserting zeros across the height and width dimensions.
        In the literature, this parameter is sometimes mentioned as input stride or dilation.
    act : activation function
        The activation function of this layer.
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
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=(3, 3), rate=2, act=tf.identity, padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
            W_init_args=None, b_init_args=None, name='atrou2d'
    ):

        super(AtrousConv2dLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "AtrousConv2dLayer %s: n_filter:%d filter_size:%s rate:%d pad:%s act:%s" %
            (name, n_filter, filter_size, rate, padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        with tf.variable_scope(name):
            shape = [filter_size[0], filter_size[1], int(self.inputs.get_shape()[-1]), n_filter]
            filters = tf.get_variable(
                name='filter', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            if b_init:
                b = tf.get_variable(
                    name='b', shape=(n_filter), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding) + b)
            else:
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding))

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([filters, b])
        else:
            self.all_params.append(filters)


def deconv2d_bilinear_upsampling_initializer(shape):
    """Returns the initializer that can be passed to DeConv2dLayer for initializ ingthe
    weights in correspondence to channel-wise bilinear up-sampling.
    Used in segmentation approaches such as [FCN](https://arxiv.org/abs/1605.06211)

    Parameters
    ----------
    shape : tuple of int
        The shape of the filters, [height, width, output_channels, in_channels].
        It must match the shape passed to DeConv2dLayer.

    Returns
    -------
    ``tf.constant_initializer``
        A constant initializer with weights set to correspond to per channel bilinear upsampling
        when passed as W_int in DeConv2dLayer

    Examples
    --------
    - Upsampling by a factor of 2, ie e.g 100->200
    >>> rescale_factor = 2
    >>> filter_size = (2 * rescale_factor - rescale_factor % 2) #Corresponding bilinear filter size
    >>> num_in_channels = 3
    >>> num_out_channels = 3
    >>> deconv_filter_shape = (filter_size, filter_size, num_out_channels, num_in_channels)
    >>> x = tf.placeholder(tf.float32, (1, imsize, imsize, num_channels))
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=filter_shape)
    >>> net = tl.layers.DeConv2dLayer(net,
    ...                    shape=filter_shape,
    ...                    output_shape=(1, imsize*rescale_factor, imsize*rescale_factor, num_out_channels),
    ...                    strides=(1, rescale_factor, rescale_factor, 1),
    ...                    W_init=bilinear_init,
    ...                    padding='SAME',
    ...                    act=tf.identity, name='g/h1/decon2d')

    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')
    if shape[3] < shape[2]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels '
        )

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    bilinear_weights_init = tf.constant_initializer(value=weights, dtype=LayersConfig.tf_dtype)  # dtype=tf.float32)
    return bilinear_weights_init


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
        Default is 'NWC' as it is a 1D CNN.
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
    >>> n = DenseLayer(n, 2, tf.identity, name='o')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=5, stride=1, dilation_rate=1, act=tf.identity, padding='SAME',
            data_format="channels_last", W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0), W_init_args=None, b_init_args=None, name='conv1d'
    ):

        super(Conv1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "Conv1d %s: n_filter:%d filter_size:%s stride:%d pad:%s act:%s dilation_rate:%d" %
            (name, n_filter, filter_size, stride, padding, act.__name__, dilation_rate)
        )

        self.inputs = prev_layer.outputs
        if tf.__version__ > '1.3':
            con1d = tf.layers.Conv1D(
                filters=n_filter, kernel_size=filter_size, strides=stride, padding=padding, data_format=data_format,
                dilation_rate=dilation_rate, activation=act, use_bias=(True if b_init else False),
                kernel_initializer=W_init, bias_initializer=b_init, name=name
            )
            # con1d.dtype = LayersConfig.tf_dtype   # unsupport, it will use the same dtype of inputs
            self.outputs = con1d(self.inputs)
            new_variables = con1d.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            self.all_layers.append(self.outputs)
            self.all_params.extend(new_variables)
        else:
            raise RuntimeError("please update TF > 1.3 or downgrade TL < 1.8.4")
        # if W_init_args is None:
        #     W_init_args = {}
        # if b_init_args is None:
        #     b_init_args = {}
        # data_format='HWC'
        # return Conv1dLayer(

    #     prev_layer=prev_layer,
    #     act=act,
    #     shape=(filter_size, int(prev_layer.outputs.get_shape()[-1]), n_filter),
    #     stride=stride,
    #     dilation_rate=dilation_rate,
    #     padding=padding,
    #     data_format=data_format,
    #     W_init=W_init,
    #     b_init=b_init,
    #     W_init_args=W_init_args,
    #     b_init_args=b_init_args,
    #     name=name,
    # )


# TODO: DeConv1d


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
    data_format : str
        "NHWC" or "NCHW", default is "NHWC" (for TF < 1.5).
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
            act=tf.identity,
            padding='SAME',
            dilation_rate=(1, 1),
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            name='conv2d',
    ):
        # if W_init_args is None:
        #     W_init_args = {}
        # if b_init_args is None:
        #     b_init_args = {}
        #
        # if len(strides) != 2:
        #     raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")
        #
        # try:
        #     pre_channel = int(layer.outputs.get_shape()[-1])
        # except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        #     pre_channel = 1
        #     logging.info("[warnings] unknow input channels, set to 1")
        # return Conv2dLayer(
        #     layer,
        #     act=act,
        #     shape=(filter_size[0], filter_size[1], pre_channel, n_filter),  # 32 features for each 5x5 patch
        #     strides=(1, strides[0], strides[1], 1),
        #     padding=padding,
        #     W_init=W_init,
        #     W_init_args=W_init_args,
        #     b_init=b_init,
        #     b_init_args=b_init_args,
        #     use_cudnn_on_gpu=use_cudnn_on_gpu,
        #     data_format=data_format,
        #     name=name)

        super(Conv2d, self).__init__(prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        if tf.__version__ > '1.5':
            logging.info(
                "Conv2d %s: n_filter:%d filter_size:%s strides:%s pad:%s act:%s" %
                (self.name, n_filter, str(filter_size), str(strides), padding, act.__name__)
            )
            # with tf.variable_scope(name) as vs:
            conv2d = tf.layers.Conv2D(
                # inputs=self.inputs,
                filters=n_filter,
                kernel_size=filter_size,
                strides=strides,
                padding=padding,
                data_format='channels_last',
                dilation_rate=dilation_rate,
                activation=act,
                use_bias=(False if b_init is None else True),
                kernel_initializer=W_init,  #None,
                bias_initializer=b_init,  #f.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                trainable=True,
                name=name,
                # reuse=None,
            )
            self.outputs = conv2d(self.inputs)
            new_variables = conv2d.weights  #trainable_variables #tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            self.all_layers.append(self.outputs)
            self.all_params.extend(new_variables)
        else:
            if len(strides) != 2:
                raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")
            try:
                pre_channel = int(prev_layer.outputs.get_shape()[-1])
            except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
                pre_channel = 1
                logging.info("[warnings] unknow input channels, set to 1")
            shape = (filter_size[0], filter_size[1], pre_channel, n_filter)  # 32 features for each 5x5 patch
            if data_format in [None, 'NHWC', 'channels_last']:
                strides = (1, strides[0], strides[1], 1)
            elif data_format in ['NCHW', 'channels']:
                strides = (1, 1, strides[0], strides[1])
            else:
                raise ValueError("`data_format` should be among 'NHWC', 'channels_last', 'NCHW', 'channels_first'.")

            logging.info(
                "Conv2d %s: shape:%s strides:%s pad:%s act:%s" %
                (self.name, str(shape), str(strides), padding, act.__name__)
            )

            with tf.variable_scope(name):
                W = tf.get_variable(
                    name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
                )
                if b_init:
                    b = tf.get_variable(
                        name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype,
                        **b_init_args
                    )
                    self.outputs = act(
                        tf.nn.conv2d(
                            self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                            data_format=data_format
                        ) + b
                    )
                else:
                    self.outputs = act(
                        tf.nn.conv2d(
                            self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                            data_format=data_format
                        )
                    )

            self.all_layers.append(self.outputs)
            if b_init:
                self.all_params.extend([W, b])
            else:
                self.all_params.append(W)


class DeConv2d(Layer):
    """Simplified version of :class:`DeConv2dLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    out_size : tuple of int
        Require if TF version < 1.3, (height, width) of output.
    strides : tuple of int
        The stride step (height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    batch_size : int or None
        Require if TF < 1.3, int or None.
        If None, try to find the `batch_size` from the first dim of net.outputs (you should define the `batch_size` in the input placeholder).
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer (For TF < 1.3).
    b_init_args : dictionary
        The arguments for the bias vector initializer (For TF < 1.3).
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', n_out_channel='n_filter', end_support_version=1.9
    )  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            out_size=(30, 30),  # remove
            strides=(2, 2),
            padding='SAME',
            batch_size=None,  # remove
            act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,  # remove
            b_init_args=None,  # remove
            name='decnn2d'
    ):
        super(DeConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DeConv2d %s: n_filters:%s strides:%s pad:%s act:%s" %
            (name, str(n_filter), str(strides), padding, act.__name__)
        )

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2, DeConv2d and DeConv2dLayer are different.")

        if tf.__version__ > '1.3':
            self.inputs = prev_layer.outputs
            # scope_name = tf.get_variable_scope().name
            conv2d_transpose = tf.layers.Conv2DTranspose(
                filters=n_filter, kernel_size=filter_size, strides=strides, padding=padding, activation=act,
                kernel_initializer=W_init, bias_initializer=b_init, name=name
            )
            self.outputs = conv2d_transpose(self.inputs)
            new_variables = conv2d_transpose.weights  # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            self.all_layers.append(self.outputs)
            self.all_params.extend(new_variables)
        else:
            raise RuntimeError("please update TF > 1.3 or downgrade TL < 1.8.4")
            # if batch_size is None:
            #     #     batch_size = tf.shape(net.outputs)[0]
            #     fixed_batch_size = prev_layer.outputs.get_shape().with_rank_at_least(1)[0]
            #     if fixed_batch_size.value:
            #         batch_size = fixed_batch_size.value
            #     else:
            #         from tensorflow.python.ops import array_ops
            #         batch_size = array_ops.shape(prev_layer.outputs)[0]
            # return DeConv2dLayer(
            #     prev_layer=prev_layer,
            #     act=act,
            #     shape=(filter_size[0], filter_size[1], n_filter, int(prev_layer.outputs.get_shape()[-1])),
            #     output_shape=(batch_size, int(out_size[0]), int(out_size[1]), n_filter),
            #     strides=(1, strides[0], strides[1], 1),
            #     padding=padding,
            #     W_init=W_init,
            #     b_init=b_init,
            #     W_init_args=W_init_args,
            #     b_init_args=b_init_args,
            #     name=name)


class DeConv3d(Layer):
    """Simplified version of The :class:`DeConv3dLayer`, see `tf.contrib.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d_transpose>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (depth, height, width).
    stride : tuple of int
        The stride step (depth, height, width).
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer, n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
            name='decnn3d'
    ):

        super(DeConv3d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DeConv3d %s: n_filters:%s strides:%s pad:%s act:%s" %
            (name, str(n_filter), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name) as vs:
            nn = tf.layers.Conv3DTranspose(
                filters=n_filter,
                kernel_size=filter_size,
                strides=strides,
                padding=padding,
                activation=act,
                kernel_initializer=W_init,
                bias_initializer=b_init,
                name=None,
            )
            self.outputs = nn(self.inputs)
            new_variables = nn.weights  # tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers.append(self.outputs)
        self.all_params.extend(new_variables)


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * depth_multiplier).

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    filter_size : tuple of int
        The filter size (height, width).
    stride : tuple of int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    dilation_rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
    depth_multiplier : int
        The number of channels to expand to.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip bias.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = InputLayer(x, name='input')
    >>> net = Conv2d(net, 32, (3, 3), (2, 2), b_init=None, name='cin')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bnin')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (1, 1), b_init=None, name='cdw1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn11')
    >>> net = Conv2d(net, 64, (1, 1), (1, 1), b_init=None, name='c1')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn12')
    ...
    >>> net = DepthwiseConv2d(net, (3, 3), (2, 2), b_init=None, name='cdw2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn21')
    >>> net = Conv2d(net, 128, (1, 1), (1, 1), b_init=None, name='c2')
    >>> net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn22')

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """ # # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            shape=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='SAME',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='depthwise_conv2d',
    ):
        super(DepthwiseConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "DepthwiseConv2d %s: shape:%s strides:%s pad:%s act:%s" %
            (name, str(shape), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknown input channels, set to 1")

        shape = [shape[0], shape[1], pre_channel, depth_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        assert len(strides) == 4, "len(strides) should be 4."

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_depthwise2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )  # [filter_height, filter_width, in_channels, depth_multiplier]
            if b_init:
                b = tf.get_variable(
                    name='b_depthwise2d', shape=(pre_channel * depth_multiplier), initializer=b_init,
                    dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=dilation_rate) + b
                )
            else:
                self.outputs = act(
                    tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=dilation_rate)
                )

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class SeparableConv1d(Layer):
    """The :class:`SeparableConv1d` class is a 1D depthwise separable convolutional layer, see `tf.layers.separable_conv1d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv1d>`__.

    This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The dimensionality of the output space (i.e. the number of filters in the convolution).
    filter_size : int
        Specifying the spatial dimensions of the filters. Can be a single integer to specify the same value for all spatial dimensions.
    strides : int
        Specifying the stride of the convolution. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding : str
        One of "valid" or "same" (case-insensitive).
    data_format : str
        One of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
    dilation_rate : int
        Specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier.
    depthwise_init : initializer
        for the depthwise convolution kernel.
    pointwise_init : initializer
        For the pointwise convolution kernel.
    b_init : initializer
        For the bias vector. If None, ignore bias in the pointwise part only.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=100,
            filter_size=3,
            strides=1,
            act=tf.identity,
            padding='valid',
            data_format='channels_last',
            dilation_rate=1,
            depth_multiplier=1,
            # activation=None,
            # use_bias=True,
            depthwise_init=None,
            pointwise_init=None,
            b_init=tf.zeros_initializer(),
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # W_init=tf.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            name='seperable1d',
    ):
        # if W_init_args is None:
        #     W_init_args = {}
        # if b_init_args is None:
        #     b_init_args = {}

        super(SeparableConv1d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "SeparableConv1d  %s: n_filter:%d filter_size:%s filter_size:%s depth_multiplier:%d act:%s" %
            (self.name, n_filter, str(filter_size), str(strides), depth_multiplier, act.__name__)
        )

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name) as vs:
            nn = tf.layers.SeparableConv1D(
                filters=n_filter,
                kernel_size=filter_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                depth_multiplier=depth_multiplier,
                activation=act,
                use_bias=(True if b_init is not None else False),
                depthwise_initializer=depthwise_init,
                pointwise_initializer=pointwise_init,
                bias_initializer=b_init,
                # depthwise_regularizer=None,
                # pointwise_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # depthwise_constraint=None,
                # pointwise_constraint=None,
                # bias_constraint=None,
                trainable=True,
                name=None
            )
            self.outputs = nn(self.inputs)
            new_variables = nn.weights

        self.all_layers.append(self.outputs)
        self.all_params.extend(new_variables)


class SeparableConv2d(Layer):
    """The :class:`SeparableConv2d` class is a 2D depthwise separable convolutional layer, see `tf.layers.separable_conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d>`__.

    This layer performs a depthwise convolution that acts separately on channels, followed by a pointwise convolution that mixes channels.
    While :class:`DepthwiseConv2d` performs depthwise convolution only, which allow us to add batch normalization between depthwise and pointwise convolution.

    Parameters
    ------------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The dimensionality of the output space (i.e. the number of filters in the convolution).
    filter_size : tuple/list of 2 int
        Specifying the spatial dimensions of the filters. Can be a single integer to specify the same value for all spatial dimensions.
    strides : tuple/list of 2 int
        Specifying the strides of the convolution. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding : str
        One of "valid" or "same" (case-insensitive).
    data_format : str
        One of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width).
    dilation_rate : integer or tuple/list of 2 int
        Specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier.
    depthwise_init : initializer
        for the depthwise convolution kernel.
    pointwise_init : initializer
        For the pointwise convolution kernel.
    b_init : initializer
        For the bias vector. If None, ignore bias in the pointwise part only.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=100,
            filter_size=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='valid',
            data_format='channels_last',
            dilation_rate=(1, 1),
            depth_multiplier=1,
            # activation=None,
            # use_bias=True,
            depthwise_init=None,
            pointwise_init=None,
            b_init=tf.zeros_initializer(),
            # depthwise_regularizer=None,
            # pointwise_regularizer=None,
            # bias_regularizer=None,
            # activity_regularizer=None,
            # depthwise_constraint=None,
            # pointwise_constraint=None,
            # W_init=tf.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            name='seperable',
    ):
        # if W_init_args is None:
        #     W_init_args = {}
        # if b_init_args is None:
        #     b_init_args = {}

        super(SeparableConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "SeparableConv2d  %s: n_filter:%d filter_size:%s filter_size:%s depth_multiplier:%d act:%s" %
            (self.name, n_filter, str(filter_size), str(strides), depth_multiplier, act.__name__)
        )

        self.inputs = prev_layer.outputs

        with tf.variable_scope(name) as vs:
            nn = tf.layers.SeparableConv2D(
                filters=n_filter,
                kernel_size=filter_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                depth_multiplier=depth_multiplier,
                activation=act,
                use_bias=(True if b_init is not None else False),
                depthwise_initializer=depthwise_init,
                pointwise_initializer=pointwise_init,
                bias_initializer=b_init,
                # depthwise_regularizer=None,
                # pointwise_regularizer=None,
                # bias_regularizer=None,
                # activity_regularizer=None,
                # depthwise_constraint=None,
                # pointwise_constraint=None,
                # bias_constraint=None,
                trainable=True,
                name=None
            )
            self.outputs = nn(self.inputs)
            new_variables = nn.weights
            # new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers.append(self.outputs)
        self.all_params.extend(new_variables)


class GroupConv2d(Layer):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

    Parameters
    --------------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : int
        The filter size.
    stride : int
        The stride step.
    n_group : int
        The number of groups.
    act : activation function
        The activation function of this layer.
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
    name : str
        A unique layer name.
    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(2, 2),
            n_group=2,
            act=tf.identity,
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='groupconv',
    ):  # Windaway

        super(GroupConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "GroupConv2d %s: n_filter:%d size:%s strides:%s n_group:%d pad:%s act:%s" %
            (name, n_filter, str(filter_size), str(strides), n_group, padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strides[0], strides[1], 1], padding=padding)
        channels = int(self.inputs.get_shape()[-1])

        with tf.variable_scope(name):
            We = tf.get_variable(
                name='W', shape=[filter_size[0], filter_size[1], channels / n_group, n_filter], initializer=W_init,
                dtype=LayersConfig.tf_dtype, trainable=True, **W_init_args
            )
            if b_init:
                bi = tf.get_variable(
                    name='b', shape=n_filter, initializer=b_init, dtype=LayersConfig.tf_dtype, trainable=True,
                    **b_init_args
                )
            if n_group == 1:
                conv = groupConv(self.inputs, We)
            else:
                inputGroups = tf.split(axis=3, num_or_size_splits=n_group, value=self.inputs)
                weightsGroups = tf.split(axis=3, num_or_size_splits=n_group, value=We)
                convGroups = [groupConv(i, k) for i, k in zip(inputGroups, weightsGroups)]
                conv = tf.concat(axis=3, values=convGroups)
            if b_init:
                conv = tf.add(conv, bi, name='add')

            self.outputs = act(conv)
        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([We, bi])
        else:
            self.all_params.append(We)


# Alias
AtrousConv1dLayer = atrous_conv1d
# Conv1d = conv1d
# Conv2d = conv2d
# DeConv2d = deconv2d
