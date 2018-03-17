# -*- coding: utf-8 -*-
from .core import *
from .. import _logging as logging
import tensorflow as tf

__all__ = [
    'BinaryDenseLayer',
    'BinaryConv2d',
    'SignLayer',
    'ScaleLayer',
]


@tf.RegisterGradient("TL_Sign_QuantizeGrad")
def _quantize_grad(op, grad):
    """Clip and binarize tensor using the straight through estimator (STE) for the gradient. """
    return tf.clip_by_value(tf.identity(grad), -1, 1)


def quantize(x):
    # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
    with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
        return tf.sign(x)


class BinaryDenseLayer(Layer):
    """The :class:`BinaryDenseLayer` class is a binary fully connected layer, which weights are either -1 or 1 while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`SignLayer` after :class:`BatchNormLayer`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=tf.identity,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='binary_dense',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        logging.info("BinaryDenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args)
            # W = tl.act.sign(W)    # dont update ...
            W = quantize(W)
            # W = tf.Variable(W)
            # print(W)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
                # self.outputs = act(xnor_gemm(self.inputs, W) + b) # TODO
            else:
                self.outputs = act(tf.matmul(self.inputs, W))
                # self.outputs = act(xnor_gemm(self.inputs, W)) # TODO

        self.all_layers.append(self.outputs)
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class BinaryConv2d(Layer):
    """
    The :class:`BinaryConv2d` class is a 2D binary CNN layer, which weights are either -1 or 1 while inferencing.

    Note that, the bias vector would not be binarized.

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
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
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

    Examples
    ---------
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.BinaryConv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='bcnn1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn1')
    ...
    >>> net = tl.layers.SignLayer(net)
    >>> net = tl.layers.BinaryConv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='bcnn2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn2')

    """

    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='SAME',
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            # act=tf.identity,
            # shape=(5, 5, 1, 100),
            # strides=(1, 1, 1, 1),
            # padding='SAME',
            # W_init=tf.truncated_normal_initializer(stddev=0.02),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            # use_cudnn_on_gpu=None,
            # data_format=None,
            name='binary_cnn2d',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        if act is None:
            act = tf.identity
        logging.info("BinaryConv2d %s: n_filter:%d filter_size:%s strides:%s pad:%s act:%s" % (self.name, n_filter, str(filter_size), str(strides), padding,
                                                                                               act.__name__))

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")
        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")
        shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
        strides = (1, strides[0], strides[1], 1)
        with tf.variable_scope(name):
            W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args)
            W = quantize(W)
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                self.outputs = act(
                    tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class SignLayer(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            name='sign',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("SignLayer  %s" % (self.name))
        with tf.variable_scope(name):
            # self.outputs = tl.act.sign(self.inputs)
            self.outputs = quantize(self.inputs)

        self.all_layers.append(self.outputs)


class ScaleLayer(Layer):
    """The :class:`AddScaleLayer` class is for multipling a trainble scale value to the layer outputs. Usually be used on the output of binary net.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    init_scale : float
        The initial value for the scale factor.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            init_scale=0.05,
            name='scale',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("ScaleLayer  %s: init_scale: %f" % (self.name, init_scale))
        with tf.variable_scope(name):
            # scale = tf.get_variable(name='scale_factor', init, trainable=True, )
            scale = tf.get_variable("scale", shape=[1], initializer=tf.constant_initializer(value=init_scale))
            self.outputs = self.inputs * scale

        self.all_layers.append(self.outputs)
        self.all_params.append(scale)
