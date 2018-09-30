#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

from tests.utils import CustomTestCase


class Network_Sequential_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            cls.model = tl.networks.Sequential(name="My_Sequential_2D_Network")

            cls.model.add(tl.layers.ReshapeLayer(shape=(-1, 16, 16, 1), name="reshape_layer_1"))

            cls.model.add(
                tl.layers.
                UpSampling2dLayer(size=(2, 2), is_scale=True, method=0, align_corners=True, name="upsample2d_layer_2")
            )
            cls.model.add(
                tl.layers.DownSampling2dLayer(
                    size=(2, 2), is_scale=True, method=0, align_corners=True, name="downsample2d_layer_2"
                )
            )
            cls.model.add(tl.layers.GaussianNoiseLayer(mean=0.0, stddev=1.0, name='noise_layer_2'))
            cls.model.add(
                tl.layers.LocalResponseNormLayer(depth_radius=5, bias=1., alpha=1., beta=.5, name='LRN_layer_2')
            )
            cls.model.add(tl.layers.BatchNormLayer(decay=0.9, epsilon=1e-5, act=None, name='batchnorm_layer_2'))
            cls.model.add(tl.layers.InstanceNormLayer(epsilon=1e-5, act=None, name='instance_norm_layer_2'))
            cls.model.add(
                tl.layers.LayerNormLayer(
                    center=True,
                    scale=True,
                    begin_norm_axis=1,
                    begin_params_axis=-1,
                    act=None,
                    name='layernorm_layer_2'
                )
            )
            cls.model.add(tl.layers.SwitchNormLayer(epsilon=1e-5, act=None, name='switchnorm_layer_2'))

            cls.model.add(tl.layers.PadLayer(padding=[[0, 0], [4, 4], [3, 3], [0, 0]], name='pad_layer_3'))
            cls.model.add(tl.layers.ZeroPad2d(padding=2, name='zeropad2d_layer_3-1'))
            cls.model.add(tl.layers.ZeroPad2d(padding=(2, 2), name='zeropad2d_layer_3-2'))
            cls.model.add(tl.layers.ZeroPad2d(padding=((3, 3), (4, 4)), name='zeropad2d_layer_3-3'))
            cls.model.add(tl.layers.ScaleLayer(init_scale=2., name='scale_layer_3'))

            cls.model.add(
                tl.layers.AtrousConv2dLayer(
                    n_filter=32, filter_size=(3, 3), rate=2, padding='SAME', act=tf.nn.relu, name='atrous_2d_layer_4'
                )
            )

            cls.model.add(
                tl.layers.AtrousConv2dLayer(
                    n_filter=32,
                    filter_size=(3, 3),
                    rate=2,
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='atrous_2d_layer_5'
                )
            )

            cls.model.add(
                tl.layers.AtrousDeConv2dLayer(
                    shape=(3, 3, 32, 32), rate=2, padding='SAME', act=tf.nn.relu, name='atrous_2d_transpose_6'
                )
            )

            cls.model.add(
                tl.layers.AtrousDeConv2dLayer(
                    shape=(3, 3, 32, 32),
                    rate=2,
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='atrous_2d_transpose_7'
                )
            )

            cls.model.add(
                tl.layers.BinaryConv2d(
                    n_filter=32,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='binary_conv2d_layer_8'
                )
            )

            cls.model.add(
                tl.layers.BinaryConv2d(
                    n_filter=32,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='binary_conv2d_layer_9'
                )
            )
            '''
            def deformable_conv_layer():
                offset_conv_layer = tl.layers.Conv2d(18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset2')
                deformable_conv = tl.layers.DeformableConv2d(
                    offset_conv_layer, 64, (3, 3), act=tf.nn.relu, name='deformable2'
                )

                return deformable_conv
            '''

            cls.model.add(
                tl.layers.DepthwiseConv2d(
                    shape=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    depth_multiplier=1,
                    act=tf.nn.relu,
                    name='depthwise_conv2d_layer_10'
                )
            )

            cls.model.add(
                tl.layers.DepthwiseConv2d(
                    shape=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    dilation_rate=(1, 1),
                    depth_multiplier=1,
                    b_init=None,
                    act=tf.nn.relu,
                    name='depthwise_conv2d_layer_11'
                )
            )

            cls.model.add(
                tl.layers.DorefaConv2d(
                    n_filter=32,
                    filter_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    bitW=1,
                    bitA=3,
                    act=tf.nn.relu,
                    name='dorefa_conv2d_layer_12'
                )
            )

            cls.model.add(
                tl.layers.DorefaConv2d(
                    n_filter=32,
                    filter_size=(3, 3),
                    strides=(1, 1),
                    padding='SAME',
                    bitW=1,
                    bitA=3,
                    b_init=None,
                    act=tf.nn.relu,
                    name='dorefa_conv2d_layer_13'
                )
            )

            cls.model.add(
                tl.layers.Conv2dLayer(
                    shape=(5, 5, 32, 16),
                    strides=(1, 1, 1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='expert_conv2d_layer_14'
                )
            )

            cls.model.add(
                tl.layers.Conv2dLayer(
                    shape=(5, 5, 16, 8),
                    strides=(1, 1, 1, 1),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='expert_conv2d_layer_15'
                )
            )

            cls.model.add(
                tl.layers.DeConv2dLayer(
                    shape=(3, 3, 8, 8),
                    strides=(1, 2, 2, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='expert_deconv2d_layer_16'
                )
            )

            cls.model.add(
                tl.layers.DeConv2dLayer(
                    shape=(3, 3, 8, 8),
                    strides=(1, 2, 2, 1),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='expert_deconv2d_layer_17'
                )
            )

            cls.model.add(
                tl.layers.GroupConv2d(
                    n_filter=32,
                    filter_size=(3, 3),
                    strides=(2, 2),
                    padding='SAME',
                    n_group=2,
                    act=tf.nn.relu,
                    name='groupconv2d_layer_18'
                )
            )

            cls.model.add(
                tl.layers.GroupConv2d(
                    n_filter=16,
                    filter_size=(3, 3),
                    strides=(2, 2),
                    padding='SAME',
                    n_group=4,
                    b_init=None,
                    act=tf.nn.relu,
                    name='groupconv2d_layer_19'
                )
            )

            cls.model.add(
                tl.layers.QuantizedConv2d(
                    n_filter=8,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    bitW=1,
                    bitA=3,
                    act=tf.nn.relu,
                    name='quantizedconv2d_layer_20'
                )
            )

            cls.model.add(
                tl.layers.QuantizedConv2d(
                    n_filter=16,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    bitW=1,
                    bitA=3,
                    b_init=None,
                    act=tf.nn.relu,
                    name='quantizedconv2d_layer_21'
                )
            )

            cls.model.add(
                tl.layers.QuantizedConv2dWithBN(
                    n_filter=8,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    bitW=1,
                    bitA=3,
                    decay=0.9,
                    act=tf.nn.relu,
                    name='quantizedconv2dwithbn_layer_22'
                )
            )

            cls.model.add(
                tl.layers.Conv2d(
                    n_filter=4,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='conv2d_layer_23'
                )
            )

            cls.model.add(
                tl.layers.Conv2d(
                    n_filter=8,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='conv2d_layer_24'
                )
            )

            cls.model.add(
                tl.layers.DeConv2d(
                    n_filter=4,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='deconv2d_layer_25'
                )
            )

            cls.model.add(
                tl.layers.DeConv2d(
                    n_filter=8,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='deconv2d_layer_26'
                )
            )

            cls.model.add(tl.layers.SubpixelConv2d(scale=2, name='subpixelconv2d_layer_27'))

            cls.model.add(
                tl.layers.Conv2d(
                    n_filter=8,
                    filter_size=(3, 3),
                    strides=(2, 2),
                    padding='SAME',
                    b_init=None,
                    act=tf.nn.relu,
                    name='conv2d_layer_28'
                )
            )

            cls.model.add(tl.layers.SubpixelConv2d(scale=2, n_out_channels=2, name='subpixelconv2d_layer_29'))

            cls.model.add(
                tl.layers.TernaryConv2d(
                    n_filter=4,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='ternaryconv2d_layer_30'
                )
            )

            cls.model.add(
                tl.layers.TernaryConv2d(
                    n_filter=8,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    b_init=None,
                    name='ternaryconv2d_layer_31'
                )
            )

            cls.model.add(
                tl.layers.SeparableConv2d(
                    n_filter=4,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    name='separableconv2d_layer_32'
                )
            )

            cls.model.add(
                tl.layers.SeparableConv2d(
                    n_filter=32,
                    filter_size=(5, 5),
                    strides=(1, 1),
                    padding='SAME',
                    act=tf.nn.relu,
                    b_init=None,
                    name='separableconv2d_layer_33'
                )
            )

            cls.model.add(tl.layers.GlobalMaxPool2d(name='globalmaxpool_2d_layer_34'))
            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 2, 2, 8], name="reshape_layer_34"))

            cls.model.add(tl.layers.GlobalMeanPool2d(name='globalmeanpool_2d_layer_35'))
            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 2, 2, 2], name="reshape_layer_35"))

            cls.model.add(
                tl.layers.MeanPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME', name='meanpool_2d_layer_36')
            )
            cls.model.add(tl.layers.TileLayer(multiples=[1, 32, 32, 8], name="tile_layer_36"))

            cls.model.add(
                tl.layers.MaxPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool_2d_layer_37')
            )

            plh = tf.placeholder(tf.float16, (100, 16, 16))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.test_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 132293)
        self.assertEqual(self.test_model.count_weights(), 132293)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 132293)

    def test_count_weights_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 60)
        self.assertEqual(len(self.test_model.get_all_weights()), 60)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 60)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 52)
        self.assertEqual(self.test_model.count_layers(), 52)
        self.assertEqual(self.model.count_layers(), 52)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float16)
                    )

                if self.test_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float16)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_layer"].outputs.shape, (100, 16, 16))
        self.assertEqual(self.test_model["input_layer"].outputs.shape, (100, 16, 16))

        self.assertEqual(self.train_model["reshape_layer_1"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["reshape_layer_1"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["upsample2d_layer_2"].outputs.shape, (100, 32, 32, 1))
        self.assertEqual(self.test_model["upsample2d_layer_2"].outputs.shape, (100, 32, 32, 1))

        self.assertEqual(self.train_model["downsample2d_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["downsample2d_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["noise_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["noise_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["LRN_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["LRN_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["batchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["batchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["instance_norm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["instance_norm_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["layernorm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["layernorm_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["switchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.test_model["switchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.train_model["pad_layer_3"].outputs.shape, (100, 24, 22, 1))
        self.assertEqual(self.test_model["pad_layer_3"].outputs.shape, (100, 24, 22, 1))

        self.assertEqual(self.train_model["zeropad2d_layer_3-1"].outputs.shape, (100, 28, 26, 1))
        self.assertEqual(self.test_model["zeropad2d_layer_3-1"].outputs.shape, (100, 28, 26, 1))

        self.assertEqual(self.train_model["zeropad2d_layer_3-2"].outputs.shape, (100, 32, 30, 1))
        self.assertEqual(self.test_model["zeropad2d_layer_3-2"].outputs.shape, (100, 32, 30, 1))

        self.assertEqual(self.train_model["zeropad2d_layer_3-3"].outputs.shape, (100, 38, 38, 1))
        self.assertEqual(self.test_model["zeropad2d_layer_3-3"].outputs.shape, (100, 38, 38, 1))

        self.assertEqual(self.train_model["scale_layer_3"].outputs.shape, (100, 38, 38, 1))
        self.assertEqual(self.test_model["scale_layer_3"].outputs.shape, (100, 38, 38, 1))

        self.assertEqual(self.train_model["atrous_2d_layer_4"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["atrous_2d_layer_4"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["atrous_2d_layer_5"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["atrous_2d_layer_5"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["atrous_2d_transpose_6"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["atrous_2d_transpose_6"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["atrous_2d_transpose_7"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["atrous_2d_transpose_7"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["binary_conv2d_layer_8"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["binary_conv2d_layer_8"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["binary_conv2d_layer_9"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["binary_conv2d_layer_9"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["depthwise_conv2d_layer_10"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["depthwise_conv2d_layer_10"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["depthwise_conv2d_layer_11"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["depthwise_conv2d_layer_11"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["dorefa_conv2d_layer_12"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["dorefa_conv2d_layer_12"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["dorefa_conv2d_layer_13"].outputs.shape, (100, 38, 38, 32))
        self.assertEqual(self.test_model["dorefa_conv2d_layer_13"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.train_model["expert_conv2d_layer_14"].outputs.shape, (100, 38, 38, 16))
        self.assertEqual(self.test_model["expert_conv2d_layer_14"].outputs.shape, (100, 38, 38, 16))

        self.assertEqual(self.train_model["expert_conv2d_layer_15"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["expert_conv2d_layer_15"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["expert_deconv2d_layer_16"].outputs.shape, (100, 75, 75, 8))
        self.assertEqual(self.test_model["expert_deconv2d_layer_16"].outputs.shape, (100, 75, 75, 8))

        self.assertEqual(self.train_model["expert_deconv2d_layer_17"].outputs.shape, (100, 149, 149, 8))
        self.assertEqual(self.test_model["expert_deconv2d_layer_17"].outputs.shape, (100, 149, 149, 8))

        self.assertEqual(self.train_model["groupconv2d_layer_18"].outputs.shape, (100, 75, 75, 32))
        self.assertEqual(self.test_model["groupconv2d_layer_18"].outputs.shape, (100, 75, 75, 32))

        self.assertEqual(self.train_model["groupconv2d_layer_19"].outputs.shape, (100, 38, 38, 16))
        self.assertEqual(self.test_model["groupconv2d_layer_19"].outputs.shape, (100, 38, 38, 16))

        self.assertEqual(self.train_model["quantizedconv2d_layer_20"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["quantizedconv2d_layer_20"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["quantizedconv2d_layer_21"].outputs.shape, (100, 38, 38, 16))
        self.assertEqual(self.test_model["quantizedconv2d_layer_21"].outputs.shape, (100, 38, 38, 16))

        self.assertEqual(self.train_model["quantizedconv2dwithbn_layer_22"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["quantizedconv2dwithbn_layer_22"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["conv2d_layer_23"].outputs.shape, (100, 38, 38, 4))
        self.assertEqual(self.test_model["conv2d_layer_23"].outputs.shape, (100, 38, 38, 4))

        self.assertEqual(self.train_model["conv2d_layer_24"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["conv2d_layer_24"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["deconv2d_layer_25"].outputs.shape, (100, 38, 38, 4))
        self.assertEqual(self.test_model["deconv2d_layer_25"].outputs.shape, (100, 38, 38, 4))

        self.assertEqual(self.train_model["deconv2d_layer_26"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["deconv2d_layer_26"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["subpixelconv2d_layer_27"].outputs.shape, (100, 76, 76, 2))
        self.assertEqual(self.test_model["subpixelconv2d_layer_27"].outputs.shape, (100, 76, 76, 2))

        self.assertEqual(self.train_model["conv2d_layer_28"].outputs.shape, (100, 38, 38, 8))
        self.assertEqual(self.test_model["conv2d_layer_28"].outputs.shape, (100, 38, 38, 8))

        self.assertEqual(self.train_model["subpixelconv2d_layer_29"].outputs.shape, (100, 76, 76, 2))
        self.assertEqual(self.test_model["subpixelconv2d_layer_29"].outputs.shape, (100, 76, 76, 2))

        self.assertEqual(self.train_model["ternaryconv2d_layer_30"].outputs.shape, (100, 76, 76, 4))
        self.assertEqual(self.test_model["ternaryconv2d_layer_30"].outputs.shape, (100, 76, 76, 4))

        self.assertEqual(self.train_model["ternaryconv2d_layer_31"].outputs.shape, (100, 76, 76, 8))
        self.assertEqual(self.test_model["ternaryconv2d_layer_31"].outputs.shape, (100, 76, 76, 8))

        self.assertEqual(self.train_model["separableconv2d_layer_32"].outputs.shape, (100, 76, 76, 4))
        self.assertEqual(self.test_model["separableconv2d_layer_32"].outputs.shape, (100, 76, 76, 4))

        self.assertEqual(self.train_model["globalmaxpool_2d_layer_34"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["globalmaxpool_2d_layer_34"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["reshape_layer_34"].outputs.shape, (100, 2, 2, 8))
        self.assertEqual(self.test_model["reshape_layer_34"].outputs.shape, (100, 2, 2, 8))

        self.assertEqual(self.train_model["globalmeanpool_2d_layer_35"].outputs.shape, (100, 8))
        self.assertEqual(self.test_model["globalmeanpool_2d_layer_35"].outputs.shape, (100, 8))

        self.assertEqual(self.train_model["reshape_layer_35"].outputs.shape, (100, 2, 2, 2))
        self.assertEqual(self.test_model["reshape_layer_35"].outputs.shape, (100, 2, 2, 2))

        self.assertEqual(self.train_model["meanpool_2d_layer_36"].outputs.shape, (100, 1, 1, 2))
        self.assertEqual(self.test_model["meanpool_2d_layer_36"].outputs.shape, (100, 1, 1, 2))

        self.assertEqual(self.train_model["tile_layer_36"].outputs.shape, (100, 32, 32, 16))
        self.assertEqual(self.test_model["tile_layer_36"].outputs.shape, (100, 32, 32, 16))

        self.assertEqual(self.train_model["maxpool_2d_layer_37"].outputs.shape, (100, 16, 16, 16))
        self.assertEqual(self.test_model["maxpool_2d_layer_37"].outputs.shape, (100, 16, 16, 16))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
