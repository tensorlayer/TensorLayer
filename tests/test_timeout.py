#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, multiprocessing
import unittest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

try:
    from tests.unittests_helper import CustomTestCase
except ImportError:
    from unittests_helper import CustomTestCase


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


def block_inception_a(inputs, scope=None, is_train=False):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding

    with tf.variable_scope(name_or_scope=scope, default_name='BlockInceptionA', values=[inputs]):
        with tf.variable_scope('Branch_0'):
            branch_0, _ = conv_module(
                inputs, n_out_channel=96, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

        with tf.variable_scope('Branch_1'):
            branch_1, _ = conv_module(
                inputs, n_out_channel=64, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=96, filter_size=(3, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_3x3'
            )

        with tf.variable_scope('Branch_2'):
            branch_2, _ = conv_module(
                inputs, n_out_channel=64, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=96, filter_size=(3, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_3x3'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=96, filter_size=(3, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_3x3'
            )

        with tf.variable_scope('Branch_3'):
            branch_3 = tl.layers.MeanPool2d(
                inputs, filter_size=(3, 3), strides=(1, 1), padding='SAME', name='AvgPool_0a_3x3'
            )

            branch_3, _ = conv_module(
                branch_3, n_out_channel=96, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x1'
            )

        return tl.layers.ConcatLayer([branch_0, branch_1, branch_2, branch_3], concat_dim=3, name='concat_layer')


def block_reduction_a(inputs, scope=None, is_train=False):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding

    with tf.variable_scope(scope, 'BlockReductionA', [inputs]):
        with tf.variable_scope('Branch_0'):
            branch_0, _ = conv_module(
                inputs, n_out_channel=384, filter_size=(3, 3), strides=(2, 2), padding='VALID', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
            )

        with tf.variable_scope('Branch_1'):
            branch_1, _ = conv_module(
                inputs, n_out_channel=192, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=224, filter_size=(3, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_3x3'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=256, filter_size=(3, 3), strides=(2, 2), padding='VALID', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
            )

        with tf.variable_scope('Branch_2'):
            branch_2 = tl.layers.MaxPool2d(inputs, (3, 3), strides=(2, 2), padding='VALID', name='MaxPool_1a_3x3')

        return tl.layers.ConcatLayer([branch_0, branch_1, branch_2], concat_dim=3, name='concat_layer')


def block_inception_b(inputs, scope=None, is_train=False):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding

    with tf.variable_scope(scope, 'BlockInceptionB', [inputs]):
        with tf.variable_scope('Branch_0'):
            branch_0, _ = conv_module(
                inputs, n_out_channel=384, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

        with tf.variable_scope('Branch_1'):
            branch_1, _ = conv_module(
                inputs, n_out_channel=192, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=224, filter_size=(1, 7), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x7'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=256, filter_size=(7, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_7x1'
            )

        with tf.variable_scope('Branch_2'):
            branch_2, _ = conv_module(
                inputs, n_out_channel=192, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=192, filter_size=(7, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_7x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=224, filter_size=(1, 7), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_1x7'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=224, filter_size=(7, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0d_7x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=256, filter_size=(1, 7), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0e_1x7'
            )

        with tf.variable_scope('Branch_3'):
            branch_3 = tl.layers.MeanPool2d(
                inputs, filter_size=(3, 3), strides=(1, 1), padding='SAME', name='AvgPool_0a_3x3'
            )

            branch_3, _ = conv_module(
                branch_3, n_out_channel=128, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x1'
            )

        return tl.layers.ConcatLayer([branch_0, branch_1, branch_2, branch_3], concat_dim=3, name='concat_layer')


def block_reduction_b(inputs, scope=None, is_train=False):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding

    with tf.variable_scope(scope, 'BlockReductionB', [inputs]):
        with tf.variable_scope('Branch_0'):
            branch_0, _ = conv_module(
                inputs, n_out_channel=192, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_0, _ = conv_module(
                branch_0, n_out_channel=192, filter_size=(3, 3), strides=(2, 2), padding='VALID', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
            )

        with tf.variable_scope('Branch_1'):
            branch_1, _ = conv_module(
                inputs, n_out_channel=256, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=256, filter_size=(1, 7), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x7'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=320, filter_size=(7, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_7x1'
            )

            branch_1, _ = conv_module(
                branch_1, n_out_channel=320, filter_size=(3, 3), strides=(2, 2), padding='VALID', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
            )

        with tf.variable_scope('Branch_2'):
            branch_2 = tl.layers.MaxPool2d(inputs, (3, 3), strides=(2, 2), padding='VALID', name='MaxPool_1a_3x3')

        return tl.layers.ConcatLayer([branch_0, branch_1, branch_2], concat_dim=3, name='concat_layer')


def block_inception_c(inputs, scope=None, is_train=False):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding

    with tf.variable_scope(scope, 'BlockInceptionC', [inputs]):
        with tf.variable_scope('Branch_0'):
            branch_0, _ = conv_module(
                inputs, n_out_channel=256, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

        with tf.variable_scope('Branch_1'):
            branch_1, _ = conv_module(
                inputs, n_out_channel=384, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_1a, _ = conv_module(
                branch_1, n_out_channel=256, filter_size=(1, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x3'
            )

            branch_1b, _ = conv_module(
                branch_1, n_out_channel=256, filter_size=(3, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_3x1'
            )

            branch_1 = tl.layers.ConcatLayer([branch_1a, branch_1b], concat_dim=3, name='concat_layer')

        with tf.variable_scope('Branch_2'):
            branch_2, _ = conv_module(
                inputs, n_out_channel=384, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=448, filter_size=(3, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_3x1'
            )

            branch_2, _ = conv_module(
                branch_2, n_out_channel=512, filter_size=(1, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0c_1x3'
            )

            branch_2a, _ = conv_module(
                branch_2, n_out_channel=256, filter_size=(1, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0d_1x3'
            )

            branch_2b, _ = conv_module(
                branch_2, n_out_channel=256, filter_size=(3, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0e_3x1'
            )

            branch_2 = tl.layers.ConcatLayer([branch_2a, branch_2b], concat_dim=3, name='concat_layer')

        with tf.variable_scope('Branch_3'):
            branch_3 = tl.layers.MeanPool2d(
                inputs, filter_size=(3, 3), strides=(1, 1), padding='SAME', name='AvgPool_0a_3x3'
            )

            branch_3, _ = conv_module(
                branch_3, n_out_channel=256, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0b_1x1'
            )

        return tl.layers.ConcatLayer([branch_0, branch_1, branch_2, branch_3], concat_dim=3, name='concat_layer')


class InceptionV4_Network(object):
    """InceptionV4 model.
    """

    def __init__(self, include_FC_head=True, flatten_output=True):

        self.include_FC_head = include_FC_head
        self.flatten_output = flatten_output

    def __call__(self, inputs, reuse=False, is_train=False):

        with tf.variable_scope("InceptionV4", reuse=reuse):

            preprocessed = inputs

            with tf.variable_scope("preprocessing"):

                max_val = tf.reduce_max(preprocessed)
                min_val = tf.reduce_min(preprocessed)

                need_int_rescale = tf.logical_and(tf.greater(max_val, 1.0), tf.greater_equal(min_val, 0.0))

                need_float_rescale = tf.logical_and(tf.less_equal(max_val, 1.0), tf.greater_equal(min_val, 0.0))

                preprocessed = tf.cond(
                    pred=need_int_rescale, true_fn=lambda: tf.subtract(tf.divide(preprocessed, 127.5), 1.0),
                    false_fn=lambda: preprocessed
                )

                preprocessed = tf.cond(
                    pred=need_float_rescale, true_fn=lambda: tf.multiply(tf.subtract(preprocessed, 0.5), 2.0),
                    false_fn=lambda: preprocessed
                )

            # Input Layers
            input_layer = tl.layers.InputLayer(preprocessed, name='input')

            # 299 x 299 x 3
            net, _ = conv_module(
                input_layer, n_out_channel=32, filter_size=(3, 3), strides=(2, 2), padding='VALID',
                batch_norm_init=None, is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
            )

            # 149 x 149 x 32
            net, _ = conv_module(
                net, n_out_channel=32, filter_size=(3, 3), strides=(1, 1), padding='VALID', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_2a_3x3'
            )

            # 147 x 147 x 32
            net, _ = conv_module(
                net, n_out_channel=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', batch_norm_init=None,
                is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_2b_3x3'
            )

            # 147 x 147 x 64
            with tf.variable_scope('Mixed_3a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = tl.layers.MaxPool2d(net, (3, 3), strides=(2, 2), padding='VALID', name='MaxPool_0a_3x3')

                with tf.variable_scope('Branch_1'):
                    branch_1, _ = conv_module(
                        net, n_out_channel=96, filter_size=(3, 3), strides=(2,
                                                                            2), padding='VALID', batch_norm_init=None,
                        is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_3x3'
                    )

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 73 x 73 x 160
            with tf.variable_scope('Mixed_4a'):
                with tf.variable_scope('Branch_0'):
                    branch_0, _ = conv_module(
                        net, n_out_channel=64, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                        is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
                    )

                    branch_0, _ = conv_module(
                        branch_0, n_out_channel=96, filter_size=(3, 3), strides=(1, 1), padding='VALID',
                        batch_norm_init=None, is_train=is_train, use_batchnorm=True, activation_fn='ReLU',
                        name='Conv2d_1a_3x3'
                    )

                with tf.variable_scope('Branch_1'):
                    branch_1, _ = conv_module(
                        net, n_out_channel=64, filter_size=(1, 1), strides=(1, 1), padding='SAME', batch_norm_init=None,
                        is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_0a_1x1'
                    )

                    branch_1, _ = conv_module(
                        branch_1, n_out_channel=64, filter_size=(1, 7), strides=(1, 1), padding='SAME',
                        batch_norm_init=None, is_train=is_train, use_batchnorm=True, activation_fn='ReLU',
                        name='Conv2d_0b_1x7'
                    )

                    branch_1, _ = conv_module(
                        branch_1, n_out_channel=64, filter_size=(7, 1), strides=(1, 1), padding='SAME',
                        batch_norm_init=None, is_train=is_train, use_batchnorm=True, activation_fn='ReLU',
                        name='Conv2d_0c_7x1'
                    )

                    branch_1, _ = conv_module(
                        branch_1, n_out_channel=96, filter_size=(3, 3), strides=(1, 1), padding='VALID',
                        batch_norm_init=None, is_train=is_train, use_batchnorm=True, activation_fn='ReLU',
                        name='Conv2d_1a_3x3'
                    )

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 71 x 71 x 192
            with tf.variable_scope('Mixed_5a'):
                with tf.variable_scope('Branch_0'):
                    # 299 x 299 x 3
                    branch_0, _ = conv_module(
                        net, n_out_channel=192, filter_size=(3, 3), strides=(2,
                                                                             2), padding='VALID', batch_norm_init=None,
                        is_train=is_train, use_batchnorm=True, activation_fn='ReLU', name='Conv2d_1a_3x3'
                    )

                with tf.variable_scope('Branch_1'):
                    branch_1 = tl.layers.MaxPool2d(net, (3, 3), strides=(2, 2), padding='VALID', name='MaxPool_1a_3x3')

                net = tl.layers.ConcatLayer([branch_0, branch_1], concat_dim=3)

            # 35 x 35 x 384
            # 4 x Inception-A blocks
            for idx in range(4):
                block_scope = 'Mixed_5' + chr(ord('b') + idx)
                net = block_inception_a(net, scope=block_scope, is_train=is_train)

            # 35 x 35 x 384
            # Reduction-A block
            net = block_reduction_a(net, scope='Mixed_6a', is_train=is_train)

            # 17 x 17 x 1024
            # 7 x Inception-B blocks
            for idx in range(7):
                block_scope = 'Mixed_6' + chr(ord('b') + idx)
                net = block_inception_b(net, scope=block_scope, is_train=is_train)

            # 17 x 17 x 1024
            # Reduction-B block
            net = block_reduction_b(net, scope='Mixed_7a', is_train=is_train)

            # 8 x 8 x 1536
            # 3 x Inception-C blocks
            for idx in range(3):
                block_scope = 'Mixed_7' + chr(ord('b') + idx)
                net = block_inception_c(net, scope=block_scope, is_train=is_train)

            if self.flatten_output and not self.include_FC_head:
                net = tl.layers.FlattenLayer(net, name='flatten')

            if self.include_FC_head:
                with tf.variable_scope("Logits", reuse=reuse):

                    # 8 x 8 x 1536
                    net = tl.layers.MeanPool2d(
                        net, filter_size=net.outputs.get_shape()[1:3], strides=(1, 1), padding='VALID',
                        name='AvgPool_1a'
                    )

                    # 1 x 1 x 1536
                    net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True, is_train=is_train, name='Dropout_1b')
                    net = tl.layers.FlattenLayer(net, name='PreLogitsFlatten')

                    # 1536
                    net, _ = dense_module(
                        net, n_units=1001, activation_fn="softmax", use_batchnorm=False, batch_norm_init=None,
                        is_train=is_train, name="Logits"
                    )

            return net


class Layer_Timeoutt_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        # with Timeout(100):

        def build_net():
            input_plh = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

            #######################################################################
            ####  =============        Model Declaration         ============= ####
            #######################################################################
            inception_v4_net = InceptionV4_Network(include_FC_head=True, flatten_output=False)

            cls.network = inception_v4_net(input_plh, reuse=False, is_train=False)
            cls.network_reuse = inception_v4_net(input_plh, reuse=True, is_train=False)

        p = multiprocessing.Process(target=build_net)
        p.start()

        # Wait for X seconds or until process finishes
        p.join(100)  # we can reduce the time, when our API are faster

        # If thread is still active
        if p.is_alive():
            print("running... let's kill it...")
            # Terminate
            p.terminate()
            p.join()
            raise Exception("timeout, too slow to build networks")

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_reuse(self):
        pass
        # self.assertEqual(self.network.count_params(), 42712937)
        # self.assertEqual(self.network_reuse.count_params(), 42712937)


if __name__ == '__main__':
    # main()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
