#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import quantize_active_overflow
from tensorlayer.layers.utils import quantize_weight_overflow

from tensorflow.python.training import moving_averages
from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = ['QuanConv2d']


class QuanConv2dWithBN(Layer):
    """The :class:`QuanConv2d` class is a binary fully connected layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
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
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, [None, 256, 256, 3])
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DorefaConv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='bcnn1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=True, name='bn1')
    ...
    >>> net = tl.layers.SignLayer(net)
    >>> net = tl.layers.DorefaConv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='bcnn2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=True, name='bn2')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            bitW=8,
            bitA=8,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            act=None,
            decay=0.9,
            epsilon=1e-5,
            is_triain=False,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            W_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            # act=None,
            # shape=(5, 5, 1, 100),
            # strides=(1, 1, 1, 1),
            # padding='SAME',
            # W_init=tf.truncated_normal_initializer(stddev=0.02),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            # use_cudnn_on_gpu=None,
            # data_format=None,
            name='quan_cnn2d_bn',
    ):
        super(QuanConv2dWithBN, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, name=name)

        logging.info(
            "QuanConv2dWithBN %s: n_filter: %d filter_size: %s strides: %s pad: %s act: %s decay: %s epsilon: %s" % (
                self.name, n_filter, str(filter_size), str(strides), padding, self.act.__name__
                if self.act is not None else 'No Activation',
                decay, epsilon
            )
        )

        self.inputs = quantize_active_overflow(self.inputs, bitA)  # Do not remove

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")

        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.warning("[warnings] unknow input channels, set to 1")

        shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
        strides = (1, strides[0], strides[1], 1)

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            
            conv = tf.nn.conv2d(
                self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                data_format=data_format
            )
            
            para_bn_shape = conv.get_shape()[-1:]


            scale_para = tf.get_variable(
                name = 'scale_para', shape = para_bn_shape, initializer = tf.ones_initializer, dtype = LayersConfig.tf_dtype, trainable = is_train)
            
            offset_para = tf.get_variable(
                name = 'offset_para', shape = para_bn_shape, initializer = tf.zeros_initializer, dtype = LayersConfig.tf_dtype, trainable = is_train)

            moving_mean = tf.get_variable(
                'moving_mean', para_bn_shape, initializer=moving_mean_init, dtype=LayersConfig.tf_dtype, trainable=False
            )

            moving_variance = tf.get_variable(
                'moving_variance',
                para_bn_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False,
            )    
            
            mean, variance = tf.nn.moments(self.inputs, axis = list(range(len(conv.get_shape()-1))))

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)
            
            if is_train:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            w_fold = _w_fold(W, scale_para, var, epsilon)
            bias_fold = _bias_fold(offset_para, scale_para, mean, var)

            W = quantize_weight_overflow(w_fold, bitW)
            
            conv_fold = tf.nn.conv2d(
                self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                data_format=data_format
            )            

            self.outputs = tf.nn.bias_add(conv_fold, bias_fold, name = 'bn_bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        self._add_params([W, scale_para, offset_para, moving_mean, moving_variance])

    def _w_fold(w, gama, var, epsilon):
        return tf.div(tf.multiply(gama, w), tf.sqrt(var + epsilon))
    
    def _bias_fold(beta, gama, mean, var):
        return tf.sub(beta, tf.div(tf.multiply(gama, mean)), tf.sqrt(var + epsilon))