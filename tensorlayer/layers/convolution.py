# -*- coding: utf-8 -*-

import tensorflow as tf

from .core import *


class Conv1dLayer(Layer):
    """
    The :class:`Conv1dLayer` class is a 1D CNN layer, see `tf.nn.convolution <https://www.tensorflow.org/api_docs/python/tf/nn/convolution>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
            act=tf.identity,
            shape=(5, 1, 5),
            stride=1,
            dilation_rate=1,
            padding='SAME',
            data_format='NWC',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='cnn1d',
    ):
        if act is None:
            act = tf.identity
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("Conv1dLayer %s: shape:%s stride:%s pad:%s act:%s" % (self.name, str(shape), str(stride), padding, act.__name__))

        with tf.variable_scope(name):
            W = tf.get_variable(name='W_conv1d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            self.outputs = tf.nn.convolution(
                self.inputs, W, strides=(stride, ), padding=padding, dilation_rate=(dilation_rate, ), data_format=data_format)  # 1.2
            if b_init:
                b = tf.get_variable(name='b_conv1d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = self.outputs + b

            self.outputs = act(self.outputs)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
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
    With TensorFlow

    >>> x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.Conv2dLayer(net,
    ...                   act = tf.nn.relu,
    ...                   shape = (5, 5, 1, 32),  # 32 features for each 5x5 patch
    ...                   strides = (1, 1, 1, 1),
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   W_init_args={},
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   b_init_args = {},
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> net = tl.layers.PoolLayer(net,
    ...                   ksize=(1, 2, 2, 1),
    ...                   strides=(1, 2, 2, 1),
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    Without TensorLayer, you can implement 2d convolution as follow.

    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )

    """

    def __init__(
            self,
            layer,
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
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if act is None:
            act = tf.identity
        logging.info("Conv2dLayer %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name):
            W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(
                    tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class DeConv2dLayer(Layer):
    """A de-convolution 2D layer.

    See `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d_transpose>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
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
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if act is None:
            act = tf.identity
        logging.info("DeConv2dLayer %s: shape:%s out_shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(output_shape), str(strides), padding,
                                                                                           act.__name__))
        # logging.info("  DeConv2dLayer: Untested")
        with tf.variable_scope(name):
            W = tf.get_variable(name='W_deconv2d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init:
                b = tf.get_variable(name='b_deconv2d', shape=(shape[-2]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding) + b)
            else:
                self.outputs = act(tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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
    b_init : initializer
        The initializer for the bias vector.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
            act=tf.identity,
            shape=(2, 2, 2, 64, 128),
            strides=(1, 2, 2, 2, 1),
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='cnn3d_layer',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if act is None:
            act = tf.identity
        logging.info("Conv3dLayer %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name):
            # W = tf.Variable(W_init(shape=shape, **W_init_args), name='W_conv')
            # b = tf.Variable(b_init(shape=[shape[-1]], **b_init_args), name='b_conv')
            W = tf.get_variable(name='W_conv3d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            b = tf.get_variable(name='b_conv3d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
            self.outputs = act(tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, name=None) + b)

        # self.outputs = act( tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, name=None) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend([W, b])


class DeConv3dLayer(Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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
    b_init : initializer
        The initializer for the bias vector.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
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
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if act is None:
            act = tf.identity
        logging.info("DeConv3dLayer %s: shape:%s out_shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(output_shape), str(strides), padding,
                                                                                           act.__name__))

        with tf.variable_scope(name):
            W = tf.get_variable(name='W_deconv3d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            b = tf.get_variable(name='b_deconv3d', shape=(shape[-2]), initializer=b_init, dtype=D_TYPE, **b_init_args)

            self.outputs = act(tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding) + b)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend([W, b])


class UpSampling2dLayer(Layer):
    """The :class:`UpSampling2dLayer` class is a up-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='upsample2d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
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
        logging.info("UpSampling2dLayer %s: is_scale:%s size:%s method:%d align_corners:%s" % (name, is_scale, size, method, align_corners))
        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(self.inputs, size=size, method=method, align_corners=align_corners)
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class DownSampling2dLayer(Layer):
    """The :class:`DownSampling2dLayer` class is down-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='downsample2d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
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
        logging.info("DownSampling2dLayer %s: is_scale:%s size:%s method:%d, align_corners:%s" % (name, is_scale, size, method, align_corners))
        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(self.inputs, size=size, method=method, align_corners=align_corners)
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class DeformableConv2d(Layer):
    """The :class:`DeformableConv2d` class is a 2D
    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    # >>> net = tl.layers.InputLayer(x, name='input_layer')
    # >>> offset_1 = tl.layers.Conv2dLayer(layer=net, act=act, shape=(3, 3, 3, 18), strides=(1, 1, 1, 1),padding='SAME', name='offset_layer1')
    # >>> net = tl.layers.DeformableConv2dLayer(layer=net, act=act, offset_layer=offset_1, shape=(3, 3, 3, 32),  name='deformable_conv_2d_layer1')
    # >>> offset_2 = tl.layers.Conv2dLayer(layer=net, act=act, shape=(3, 3, 32, 18), strides=(1, 1, 1, 1), padding='SAME', name='offset_layer2')
    # >>> net = tl.layers.DeformableConv2dLayer(layer=net, act=act, offset_layer=offset_2, shape=(3, 3, 32, 64), name='deformable_conv_2d_layer2')
    def __init__(
            self,
            layer,
            offset_layer=None,
            # shape=(3, 3, 1, 100),
            n_filter=32,
            filter_size=(3, 3),
            act=tf.identity,
            name='deformable_conv_2d',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None):
        if tf.__version__ < "1.4":
            raise Exception("Deformable CNN layer requires tensrflow 1.4 or higher version | current version %s" % tf.__version__)

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
                ],
                axis=-1)
            coords = tf.tile(coords, [channel, 1, 1, 1, 1])

            mapped_vals = _tf_batch_map_coordinates(inputs, coords)
            # (b*c, h, w, n) --> (b, h, w, n, c)
            mapped_vals = _to_b_h_w_n_c(mapped_vals, [batch_size, input_h, input_w, kernel_n, channel])

            return mapped_vals

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.offset_layer = offset_layer
        if act is None:
            act = tf.identity
        logging.info("DeformableConv2d %s: n_filter: %d, filter_size: %s act:%s" % (self.name, n_filter, str(filter_size), act.__name__))

        try:
            pre_channel = int(layer.outputs.get_shape()[-1])
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
            initial_offsets = tf.stack(tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), indexing='ij'))  # initial_offsets --> (kh, kw, 2)
            initial_offsets = tf.reshape(initial_offsets, (-1, 2))  # initial_offsets --> (n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, 1, n, 2)
            initial_offsets = tf.tile(initial_offsets, [input_h, input_w, 1, 1])  # initial_offsets --> (h, w, n, 2)
            initial_offsets = tf.cast(initial_offsets, 'float32')
            grid = tf.meshgrid(
                tf.range(-int((shape[0] - 1) / 2.0), int(input_h - int((shape[0] - 1) / 2.0)), 1),
                tf.range(-int((shape[1] - 1) / 2.0), int(input_w - int((shape[1] - 1) / 2.0)), 1),
                indexing='ij')

            grid = tf.stack(grid, axis=-1)
            grid = tf.cast(grid, 'float32')  # grid --> (h, w, 2)
            grid = tf.expand_dims(grid, 2)  # grid --> (h, w, 1, 2)
            grid = tf.tile(grid, [1, 1, kernel_n, 1])  # grid --> (h, w, n, 2)
            grid_offset = grid + initial_offsets  # grid_offset --> (h, w, n, 2)

            input_deform = _tf_batch_map_offsets(self.inputs, offset, grid_offset)

            W = tf.get_variable(
                name='W_deformableconv2d', shape=[1, 1, shape[0] * shape[1], shape[-2], shape[-1]], initializer=W_init, dtype=D_TYPE, **W_init_args)

            if b_init:
                b = tf.get_variable(name='b_deformableconv2d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = tf.reshape(
                    act(tf.nn.conv3d(input_deform, W, strides=[1, 1, 1, 1, 1], padding='VALID', name=None) + b),
                    (tf.shape(self.inputs)[0], input_h, input_w, shape[-1]))
            else:
                self.outputs = tf.reshape(
                    act(tf.nn.conv3d(input_deform, W, strides=[1, 1, 1, 1, 1], padding='VALID', name=None)),
                    (tf.shape(self.inputs)[0], input_h, input_w, shape[-1]))

        # fixed
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # offset_layer
        offset_params = [osparam for osparam in offset_layer.all_params if osparam not in layer.all_params]
        offset_layers = [oslayer for oslayer in offset_layer.all_layers if oslayer not in layer.all_layers]

        self.all_params.extend(list(offset_params))
        self.all_layers.extend(list(offset_layers))
        self.all_drop.update(dict(offset_layer.all_drop))

        # this layer
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


def atrous_conv1d(
        layer,
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
    layer : :class:`Layer`
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
        layer=layer,
        act=act,
        shape=(filter_size, int(layer.outputs.get_shape()[-1]), n_filter),
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
    layer : :class:`Layer`
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

    def __init__(self,
                 layer,
                 n_filter=32,
                 filter_size=(3, 3),
                 rate=2,
                 act=tf.identity,
                 padding='SAME',
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args=None,
                 b_init_args=None,
                 name='atrou2d'):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if act is None:
            act = tf.identity
        logging.info("AtrousConv2dLayer %s: n_filter:%d filter_size:%s rate:%d pad:%s act:%s" % (self.name, n_filter, filter_size, rate, padding, act.__name__))
        with tf.variable_scope(name):
            shape = [filter_size[0], filter_size[1], int(self.inputs.get_shape()[-1]), n_filter]
            filters = tf.get_variable(name='filter', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init:
                b = tf.get_variable(name='b', shape=(n_filter), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding) + b)
            else:
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([filters, b])
        else:
            self.all_params.extend([filters])


class _SeparableConv2dLayer(Layer):  # TODO
    """The :class:`SeparableConv2dLayer` class is 2D convolution with separable filters, see `tf.layers.separable_conv2d <https://www.tensorflow.org/api_docs/python/tf/layers/separable_conv2d>`__.

    This layer has not been fully tested yet.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer with a 4D output tensor in the shape of [batch, height, width, channels].
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The strides (height, width).
        This can be a single integer if you want to specify the same value for all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding : str
        The type of padding algorithm: "SAME" or "VALID"
    data_format : str
        One of channels_last (Default) or channels_first.
        The order must match the input dimensions.
        channels_last corresponds to inputs with shapedata_format = 'NWHC' (batch, width, height, channels) while
        channels_first corresponds to inputs with shape [batch, channels, width, height].
    dilation_rate : int or tuple of ints
        The dilation rate of the convolution.
        It can be a single integer if you want to specify the same value for all spatial dimensions.
        Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
    depth_multiplier : int
        The number of depthwise convolution output channels for each input channel.
        The total number of depthwise convolution output channels will be equal to num_filters_in * depth_multiplier.
    act : activation function
        The activation function of this layer.
    use_bias : boolean
        Whether the layer uses a bias
    depthwise_initializer : initializer
        The initializer for the depthwise convolution kernel.
    pointwise_initializer : initializer
        The initializer for the pointwise convolution kernel.
    bias_initializer : initializer
        The initializer for the bias vector. If None, skip bias.
    depthwise_regularizer : regularizer
        Optional regularizer for the depthwise convolution kernel.
    pointwise_regularizer : regularizer
        Optional regularizer for the pointwise convolution kernel.
    bias_regularizer : regularizer
        Optional regularizer for the bias vector.
    activity_regularizer : regularizer
        Regularizer function for the output.
    name : str
        A unique layer name.

    """

    def __init__(self,
                 layer,
                 n_filter,
                 filter_size=5,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 act=tf.identity,
                 use_bias=True,
                 depthwise_initializer=None,
                 pointwise_initializer=None,
                 bias_initializer=tf.zeros_initializer,
                 depthwise_regularizer=None,
                 pointwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 name='atrou2d'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if tf.__version__ > "0.12.1":
            raise Exception("This layer only supports for TF 1.0+")

        bias_initializer = bias_initializer()

        logging.info("SeparableConv2dLayer %s: n_filter:%d filter_size:%s strides:%s padding:%s dilation_rate:%s depth_multiplier:%s act:%s" %
                     (self.name, n_filter, filter_size, str(strides), padding, str(dilation_rate), str(depth_multiplier), act.__name__))

        with tf.variable_scope(name) as vs:
            self.outputs = tf.layers.separable_conv2d(
                self.inputs,
                filters=n_filter,
                kernel_size=filter_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                depth_multiplier=depth_multiplier,
                activation=act,
                use_bias=use_bias,
                depthwise_initializer=depthwise_initializer,
                pointwise_initializer=pointwise_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=depthwise_regularizer,
                pointwise_regularizer=pointwise_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
            )
            # trainable=True, name=None, reuse=None)

            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


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
        raise Exception('deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels ')

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
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    bilinear_weights_init = tf.constant_initializer(value=weights, dtype=D_TYPE)  # dtype=tf.float32)
    return bilinear_weights_init


def conv1d(
        layer,
        n_filter=32,
        filter_size=5,
        stride=1,
        dilation_rate=1,
        act=tf.identity,
        padding='SAME',
        data_format="NWC",
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        name='conv1d',
):
    """Simplified version of :class:`Conv1dLayer`.

    Parameters
    ----------
    layer : :class:`Layer`
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
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name

    Returns
    -------
    :class:`Layer`
        A :class:`Conv1dLayer` object.

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

    if W_init_args is None:
        W_init_args = {}
    if b_init_args is None:
        b_init_args = {}

    return Conv1dLayer(
        layer=layer,
        act=act,
        shape=(filter_size, int(layer.outputs.get_shape()[-1]), n_filter),
        stride=stride,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format,
        W_init=W_init,
        b_init=b_init,
        W_init_args=W_init_args,
        b_init_args=b_init_args,
        name=name,
    )


def conv2d(
        layer,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=tf.identity,
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        use_cudnn_on_gpu=None,
        data_format=None,
        name='conv2d',
):
    """Simplified version of :class:`Conv2dLayer`.

    Parameters
    ----------
    layer : :class:`Layer`
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
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Returns
    -------
    :class:`Layer`
        A :class:`Conv2dLayer` object.

    Examples
    --------
    >>> net = InputLayer(x, name='inputs')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_1')
    >>> net = Conv2d(net, 64, (3, 3), act=tf.nn.relu, name='conv1_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_1')
    >>> net = Conv2d(net, 128, (3, 3), act=tf.nn.relu, name='conv2_2')
    >>> net = MaxPool2d(net, (2, 2), name='pool2')

    """

    if W_init_args is None:
        W_init_args = {}
    if b_init_args is None:
        b_init_args = {}

    if len(strides) != 2:
        raise ValueError("len(strides) should be 2, Conv2d and Conv2dLayer are different.")

    try:
        pre_channel = int(layer.outputs.get_shape()[-1])
    except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        pre_channel = 1
        logging.info("[warnings] unknow input channels, set to 1")
    return Conv2dLayer(
        layer,
        act=act,
        shape=(filter_size[0], filter_size[1], pre_channel, n_filter),  # 32 features for each 5x5 patch
        strides=(1, strides[0], strides[1], 1),
        padding=padding,
        W_init=W_init,
        W_init_args=W_init_args,
        b_init=b_init,
        b_init_args=b_init_args,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        name=name)


def deconv2d(layer,
             n_filter=32,
             filter_size=(3, 3),
             out_size=(30, 30),
             strides=(2, 2),
             padding='SAME',
             batch_size=None,
             act=tf.identity,
             W_init=tf.truncated_normal_initializer(stddev=0.02),
             b_init=tf.constant_initializer(value=0.0),
             W_init_args=None,
             b_init_args=None,
             name='decnn2d'):
    """Simplified version of :class:`DeConv2dLayer`.

    Parameters
    ----------
    layer : :class:`Layer`
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
    batch_size : int
        Require if TF version < 1.3, int or None.
        If None, try to find the `batch_size` from the first dim of net.outputs (you should define the `batch_size` in the input placeholder).
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

    Returns
    -------
    :class:`Layer`
        A :class:`DeConv2dLayer` object.

    """
    if W_init_args is None:
        W_init_args = {}
    if b_init_args is None:
        b_init_args = {}
    if act is None:
        act = tf.identity
    if len(strides) != 2:
        raise ValueError("len(strides) should be 2, DeConv2d and DeConv2dLayer are different.")
    if tf.__version__ > '1.3':
        logging.info("DeConv2d %s: n_filters:%s strides:%s pad:%s act:%s" % (name, str(n_filter), str(strides), padding, act.__name__))
        inputs = layer.outputs
        scope_name = tf.get_variable_scope().name
        if scope_name:
            whole_name = scope_name + '/' + name
        else:
            whole_name = name
        net_new = Layer(inputs, name=whole_name)
        # with tf.name_scope(name):
        with tf.variable_scope(name) as vs:
            net_new.outputs = tf.contrib.layers.conv2d_transpose(
                inputs=inputs,
                num_outputs=n_filter,
                kernel_size=filter_size,
                stride=strides,
                padding=padding,
                activation_fn=act,
                weights_initializer=W_init,
                biases_initializer=b_init,
                scope=name)
            new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        net_new.all_layers = list(layer.all_layers)
        net_new.all_params = list(layer.all_params)
        net_new.all_drop = dict(layer.all_drop)
        net_new.all_layers.extend([net_new.outputs])
        net_new.all_params.extend(new_variables)
        return net_new
    else:
        if batch_size is None:
            #     batch_size = tf.shape(net.outputs)[0]
            fixed_batch_size = layer.outputs.get_shape().with_rank_at_least(1)[0]
            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                from tensorflow.python.ops import array_ops
                batch_size = array_ops.shape(layer.outputs)[0]
        return DeConv2dLayer(
            layer=layer,
            act=act,
            shape=(filter_size[0], filter_size[1], n_filter, int(layer.outputs.get_shape()[-1])),
            output_shape=(batch_size, int(out_size[0]), int(out_size[1]), n_filter),
            strides=(1, strides[0], strides[1], 1),
            padding=padding,
            W_init=W_init,
            b_init=b_init,
            W_init_args=W_init_args,
            b_init_args=b_init_args,
            name=name)


class DeConv3d(Layer):
    """Simplified version of The :class:`DeConv3dLayer`, see `tf.contrib.layers.conv3d_transpose <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv3d_transpose>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(self,
                 layer,
                 n_filter=32,
                 filter_size=(3, 3, 3),
                 strides=(2, 2, 2),
                 padding='SAME',
                 act=tf.identity,
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0.0),
                 name='decnn3d'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("DeConv3d %s: n_filters:%s strides:%s pad:%s act:%s" % (name, str(n_filter), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            self.outputs = tf.contrib.layers.conv3d_transpose(
                num_outputs=n_filter,
                kernel_size=filter_size,
                stride=strides,
                padding=padding,
                activation_fn=act,
                weights_initializer=W_init,
                biases_initializer=b_init,
                scope=name,
            )
            new_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(new_variables)


class DepthwiseConv2d(Layer):
    """Separable/Depthwise Convolutional 2D layer, see `tf.nn.depthwise_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/tf/nn/depthwise_conv2d>`__.

    Input:
        4-D Tensor (batch, height, width, in_channels).
    Output:
        4-D Tensor (batch, new height, new width, in_channels * channel_multiplier).

    Parameters
    ------------
    layer : :class:`Layer`
        Previous layer.
    channel_multiplier : int
        The number of channels to expand to.
    filter_size : tuple of int
        The filter size (height, width).
    stride : tuple of int
        The stride step (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    rate: tuple of 2 int
        The dilation rate in which we sample input values across the height and width dimensions in atrous convolution. If it is greater than 1, then all values of strides must be 1.
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
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
    >>> n = InputLayer(x, name='in')
    >>> n = Conv2d(n, 32, (3, 3), (2, 2), act=tf.nn.relu, name='c1')
    >>> n = DepthwiseConv2d(n, 1, (3, 3), (1, 1), name='d1')
    >>> print(n.outputs.get_shape())
    ... (?, 14, 14, 32)

    References
    -----------
    - tflearn's `grouped_conv_2d <https://github.com/tflearn/tflearn/blob/3e0c3298ff508394f3ef191bcd7d732eb8860b2e/tflearn/layers/conv.py>`__
    - keras's `separableconv2d <https://keras.io/layers/convolutional/#separableconv2d>`__

    """ # # https://zhuanlan.zhihu.com/p/31551004  https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/MobileNet.py

    def __init__(
            self,
            layer,
            channel_multiplier=1,
            shape=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='SAME',
            rate=(1, 1),
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='depthwise_conv2d',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        if act is None:
            act = tf.identity

        logging.info("DepthwiseConv2d %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))
        try:
            pre_channel = int(layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")

        shape = [shape[0], shape[1], pre_channel, channel_multiplier]

        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        assert len(strides) == 4, "len(strides) should be 4."

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_depthwise2d', shape=shape, initializer=W_init, dtype=D_TYPE,
                **W_init_args)  # [filter_height, filter_width, in_channels, channel_multiplier]
            if b_init:
                b = tf.get_variable(name='b_depthwise2d', shape=(pre_channel * channel_multiplier), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=rate) + b)
            else:
                self.outputs = act(tf.nn.depthwise_conv2d(self.inputs, W, strides=strides, padding=padding, rate=rate))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class GroupConv2d(Layer):
    """The :class:`GroupConv2d` class is 2D grouped convolution, see `here <https://blog.yani.io/filter-group-tutorial/>`__.

    Parameters
    --------------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer=None,
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
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=[1, strides[0], strides[1], 1], padding=padding)
        channels = int(self.inputs.get_shape()[-1])
        with tf.variable_scope(name):
            We = tf.get_variable(
                name='W', shape=[filter_size[0], filter_size[1], channels / n_group, n_filter], initializer=W_init, dtype=D_TYPE, trainable=True, **W_init_args)
            if b_init:
                bi = tf.get_variable(name='b', shape=n_filter, initializer=b_init, dtype=D_TYPE, trainable=True, **b_init_args)
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
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([We, bi])
        else:
            self.all_params.append(We)


# Alias
AtrousConv1dLayer = atrous_conv1d
Conv1d = conv1d
Conv2d = conv2d
DeConv2d = deconv2d
DeformableConv2dLayer = DeformableConv2d
