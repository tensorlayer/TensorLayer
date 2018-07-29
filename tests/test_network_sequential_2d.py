#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

try:
    from tests.unittests_helper import CustomTestCase
except ImportError:
    from unittests_helper import CustomTestCase


class Network_Sequential_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            cls.model = tl.networks.Sequential(name="My_Seq_net")

            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 16, 16, 1], name="reshape_layer_1"))

            cls.model.add(
                tl.layers.UpSampling2dLayer(
                    size=(2, 2), is_scale=True, method=0, align_corners=True, name="upsample2d_layer_2"
                )
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
                    center=True, scale=True, begin_norm_axis=1, begin_params_axis=-1, act=None, name='layernorm_layer_2'
                )
            )
            cls.model.add(tl.layers.SwitchNormLayer(epsilon=1e-5, act=None, name='switchnorm_layer_2'))

            cls.model.add(tl.layers.PadLayer(padding=[[0, 0], [4, 4], [3, 3], [0, 0]], name='pad_layer_3'))
            cls.model.add(tl.layers.ZeroPad2d(padding=2, name='zeropad2d_layer_3-1'))
            cls.model.add(tl.layers.ZeroPad2d(padding=(2, 2), name='zeropad2d_layer_3-2'))
            cls.model.add(tl.layers.ZeroPad2d(padding=((3, 3), (4, 4)), name='zeropad2d_layer_3-3'))
            cls.model.add(tl.layers.ScaleLayer(init_scale=2., name='scale_layer_3'))

            cls.model.add(tl.layers.AtrousConv2dLayer(n_filter=32, filter_size=(3, 3), rate=2, padding='SAME', act=tf.nn.relu, name='atrous_2d_layer_4'))

            cls.model.add(tl.layers.AtrousConv2dLayer(n_filter=32, filter_size=(3, 3), rate=2, padding='SAME', b_init=None, act=tf.nn.relu, name='atrous_2d_layer_5'))

            cls.model.add(tl.layers.AtrousDeConv2dLayer(shape=(3, 3, 32, 32), output_shape=(None, 64, 64, 32), rate=2, padding='SAME', act=tf.nn.relu, name='atrous_2d_transpose_6'))

            cls.model.add(tl.layers.AtrousDeConv2dLayer(shape=(3, 3, 32, 32), output_shape=(None, 128, 128, 32), rate=2, padding='SAME', b_init=None, act=tf.nn.relu, name='atrous_2d_transpose_7'))

            plh = tf.placeholder(tf.float16, (100, 16, 16))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.model.all_drop), 0)


    def test_count_params(self):
        self.assertEqual(self.model.count_params(), 28017)

    def test_count_param_tensors(self):
        self.assertEqual(len(self.model.get_all_params()), 19)

    def test_count_layers(self):
        self.assertEqual(self.model.count_layers(), 19)

    def test_network_shapes(self):

        self.assertEqual(self.model["input_layer"].outputs.shape, (100, 16, 16))

        self.assertEqual(self.model["reshape_layer_1"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.model["upsample2d_layer_2"].outputs.shape, (100, 32, 32, 1))
        self.assertEqual(self.model["downsample2d_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["noise_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["LRN_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["batchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["instance_norm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["layernorm_layer_2"].outputs.shape, (100, 16, 16, 1))
        self.assertEqual(self.model["switchnorm_layer_2"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.model["pad_layer_3"].outputs.shape, (100, 24, 22, 1))
        self.assertEqual(self.model["zeropad2d_layer_3-1"].outputs.shape, (100, 28, 26, 1))
        self.assertEqual(self.model["zeropad2d_layer_3-2"].outputs.shape, (100, 32, 30, 1))
        self.assertEqual(self.model["zeropad2d_layer_3-3"].outputs.shape, (100, 38, 38, 1))
        self.assertEqual(self.model["scale_layer_3"].outputs.shape, (100, 38, 38, 1))

        self.assertEqual(self.model["atrous_2d_layer_4"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.model["atrous_2d_layer_5"].outputs.shape, (100, 38, 38, 32))

        self.assertEqual(self.model["atrous_2d_transpose_6"].outputs.shape, (100, 64, 64, 32))

        self.assertEqual(self.model["atrous_2d_transpose_7"].outputs.shape, (100, 128, 128, 32))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
