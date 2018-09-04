#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'activation_module',
    'conv_module',
    'dense_module',
]


def activation_module(layer, activation_fn, leaky_relu_alpha=0.2, name=None):

    act_name = name + "/activation" if name is not None else "activation"

    if activation_fn not in ["ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6", "PTReLU6", "CReLU", "ELU", "SELU",
                             "tanh", "sigmoid", "softmax", None]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    elif activation_fn == "ReLU":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.relu, name=act_name)

    elif activation_fn == "ReLU6":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.relu6, name=act_name)

    elif activation_fn == "Leaky_ReLU":
        layer = tl.layers.LambdaLayer(
            prev_layer=layer, fn=tf.nn.leaky_relu, fn_args={'alpha': leaky_relu_alpha}, name=act_name
        )

    elif activation_fn == "PReLU":
        layer = tl.layers.PReluLayer(prev_layer=layer, channel_shared=False, name=act_name)

    elif activation_fn == "PReLU6":
        layer = tl.layers.PRelu6Layer(prev_layer=layer, channel_shared=False, name=act_name)

    elif activation_fn == "PTReLU6":
        layer = tl.layers.PTRelu6Layer(prev_layer=layer, channel_shared=False, name=act_name)

    elif activation_fn == "CReLU":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.crelu, name=act_name)

    elif activation_fn == "ELU":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.elu, name=act_name)

    elif activation_fn == "SELU":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.selu, name=act_name)

    elif activation_fn == "tanh":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.tanh, name=act_name)

    elif activation_fn == "sigmoid":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.sigmoid, name=act_name)

    elif activation_fn == "softmax":
        layer = tl.layers.LambdaLayer(prev_layer=layer, fn=tf.nn.softmax, name=act_name)

    return layer


def conv_module(
        prev_layer, n_out_channel, filter_size, strides, padding, is_train=True, use_batchnorm=True, activation_fn=None,
        conv_init=tf.contrib.layers.xavier_initializer(uniform=True),
        batch_norm_init=tf.truncated_normal_initializer(mean=1.,
                                                        stddev=0.02), bias_init=tf.zeros_initializer(), name=None
):

    if activation_fn not in ["ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6", "PTReLU6", "CReLU", "ELU", "SELU",
                             "tanh", "sigmoid", "softmax", None]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    conv_name = 'conv2d' if name is None else name
    bn_name = 'batch_norm' if name is None else name + '/BatchNorm'

    layer = tl.layers.Conv2d(
        prev_layer,
        n_filter=n_out_channel,
        filter_size=filter_size,
        strides=strides,
        padding=padding,
        act=None,
        W_init=conv_init,
        b_init=None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
        name=conv_name
    )

    if use_batchnorm:

        layer = tl.layers.BatchNormLayer(layer, act=None, is_train=is_train, gamma_init=batch_norm_init, name=bn_name)

    logits = layer.outputs

    layer = activation_module(layer, activation_fn, name=conv_name)

    return layer, logits


def dense_module(
        prev_layer, n_units, is_train, use_batchnorm=True, activation_fn=None,
        dense_init=tf.contrib.layers.xavier_initializer(uniform=True),
        batch_norm_init=tf.truncated_normal_initializer(mean=1.,
                                                        stddev=0.02), bias_init=tf.zeros_initializer(), name=None
):

    if activation_fn not in ["ReLU", "ReLU6", "Leaky_ReLU", "PReLU", "PReLU6", "PTReLU6", "CReLU", "ELU", "SELU",
                             "tanh", "sigmoid", "softmax", None]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    # Flatten: Conv to FC
    if prev_layer.outputs.get_shape().__len__() != 2:  # The input dimension must be rank 2
        layer = tl.layers.FlattenLayer(prev_layer, name='flatten')

    else:
        layer = prev_layer

    layer = tl.layers.DenseLayer(
        layer,
        n_units=n_units,
        act=None,
        W_init=dense_init,
        b_init=None if use_batchnorm else bias_init,  # Not useful as the convolutions are batch normalized
        name='dense' if name is None else name
    )

    if use_batchnorm:
        layer = tl.layers.BatchNormLayer(
            layer, act=None, is_train=is_train, gamma_init=batch_norm_init, name='batch_norm'
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn)

    return layer, logits
