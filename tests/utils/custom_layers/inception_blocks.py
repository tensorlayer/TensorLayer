#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tests.utils.custom_layers.basic_layers import conv_module

__all__ = [
    'block_inception_a',
    'block_reduction_a',
    'block_inception_b',
    'block_reduction_b',
    'block_inception_c',
    'block_reduction_b',
]


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
