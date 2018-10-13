#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 100, 3))

        cls.input_layer = tl.layers.Input(name='input_layer')(x)

        cls.n1 = tl.layers.Conv2d(
            act=tf.nn.relu,
            shape=(5, 5, 3, 32),
            strides=(1, 2, 2, 1),
            padding='valid',
            W_init=tf.truncated_normal_initializer(stddev=5e-2),
            b_init=tf.constant_initializer(value=0.0),
            name='conv2dlayer'
        )(cls.input_layer)

        cls.n2 = tl.layers.Conv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), padding='valid', act=None, name='conv2d'
        )(cls.n1)

        cls.n3 = tl.layers.Conv2d(
            n_filter=32,
            filter_size=(3, 3),
            strides=(2, 2),
            padding='valid',
            act=tf.nn.relu,
            b_init=None,
            name='conv2d_no_bias'
        )(cls.n2)

        cls.n4 = tl.layers.DeConv2d(
            shape=(5, 5, 32, 32), strides=(1, 2, 2, 1), padding='valid', name='deconv2dlayer'
        )(cls.n3)

        cls.n5 = tl.layers.DeConv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), padding='valid', name='DeConv2d'
        )(cls.n4)

        cls.n6 = tl.layers.DepthwiseConv2d(
            shape=(3, 3), strides=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise'
        )(cls.n5)

        cls.n7 = tl.layers.Conv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), padding='valid', act=tf.nn.relu, name='conv2d2'
        )(cls.n6)

        cls.n8 = tl.layers.GroupConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='group')(cls.n7)

        cls.n9 = tl.layers.QuantizedConv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='valid', name='quancnn')(cls.n8)

        cls.n10 = tl.layers.UpSampling2d(
            size=(2, 2), is_scale=True, method=0, align_corners=True, name="upsample2d_layer"
        )(cls.n9)

        cls.n11 = tl.layers.UpSampling2d(
            size=(2, 2), is_scale=True, method=0, align_corners=True, name="upsample2d_layer"
        )(cls.n10)

        cls.n12 = tl.layers.SeparableConv2d(
            n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, name='seperable2d1'
        )(cls.n11)

        cls.n13 = tl.layers.TernaryConv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='valid', name='cnn2')(cls.n12)

        cls.n14 = tl.layers.AtrousDeConv2d(
            shape=(3, 3, 32, 64), rate=2, act=tf.nn.relu, name='atroustrans1', padding='valid'
        )(cls.n13)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_weights), 2)
        self.assertEqual(self.n1.count_weights(), 2432)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [48, 48, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_weights), 4)
        self.assertEqual(self.n2.count_weights(), 11680)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [23, 23, 32])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_weights), 5)
        self.assertEqual(self.n3.count_weights(), 20896)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [11, 11, 32])

    def test_layer_n4(self):

        self.assertEqual(len(self.n4.all_layers), 5)
        self.assertEqual(len(self.n4.all_weights), 7)
        self.assertEqual(self.n4.count_weights(), 46528)
        self.assertEqual(self.n4.outputs.get_shape().as_list()[1:], [25, 25, 32])

    def test_layer_n5(self):

        self.assertEqual(len(self.n5.all_layers), 6)
        self.assertEqual(len(self.n5.all_weights), 9)
        self.assertEqual(self.n5.count_weights(), 55776)
        self.assertEqual(self.n5.outputs.get_shape().as_list()[1:], [51, 51, 32])

    def test_layer_n6(self):

        self.assertEqual(len(self.n6.all_layers), 7)
        self.assertEqual(len(self.n6.all_weights), 11)
        self.assertEqual(self.n6.count_weights(), 56416)
        self.assertEqual(self.n6.outputs.get_shape().as_list()[1:], [26, 26, 64])

    def test_layer_n7(self):

        self.assertEqual(len(self.n7.all_layers), 8)
        self.assertEqual(len(self.n7.all_weights), 13)
        self.assertEqual(self.n7.count_weights(), 74880)
        self.assertEqual(self.n7.outputs.get_shape().as_list()[1:], [12, 12, 32])

    def test_layer_n8(self):

        self.assertEqual(len(self.n8.all_layers), 9)
        self.assertEqual(len(self.n8.all_weights), 15)
        self.assertEqual(self.n8.count_weights(), 79520)
        self.assertEqual(self.n8.outputs.get_shape().as_list()[1:], [6, 6, 32])

    def test_layer_n9(self):

        self.assertEqual(len(self.n9.all_layers), 10)
        self.assertEqual(len(self.n9.all_weights), 17)
        self.assertEqual(self.n9.count_weights(), 130784)
        self.assertEqual(self.n9.outputs.get_shape().as_list()[1:], [2, 2, 64])

    def test_layer_n10(self):

        self.assertEqual(len(self.n10.all_layers), 11)
        self.assertEqual(len(self.n10.all_weights), 17)
        self.assertEqual(self.n10.count_weights(), 130784)
        self.assertEqual(self.n10.outputs.get_shape().as_list()[1:], [4, 4, 64])

    def test_layer_n11(self):

        self.assertEqual(len(self.n11.all_layers), 12)
        self.assertEqual(len(self.n11.all_weights), 17)
        self.assertEqual(self.n11.count_weights(), 130784)
        self.assertEqual(self.n11.outputs.get_shape().as_list()[1:], [8, 8, 64])

    def test_layer_n12(self):

        self.assertEqual(len(self.n12.all_layers), 13)
        self.assertEqual(len(self.n12.all_weights), 20)
        self.assertEqual(self.n12.count_weights(), 133440)
        self.assertEqual(self.n12.outputs.get_shape().as_list()[1:], [6, 6, 32])

    def test_layer_n13(self):

        self.assertEqual(len(self.n13.all_layers), 14)
        self.assertEqual(len(self.n13.all_weights), 22)
        self.assertEqual(self.n13.count_weights(), 151936)
        self.assertEqual(self.n13.outputs.get_shape().as_list()[1:], [4, 4, 64])

    def test_layer_n14(self):

        self.assertEqual(len(self.n14.all_layers), 15)
        self.assertEqual(len(self.n14.all_weights), 24)
        self.assertEqual(self.n14.count_weights(), 170400)
        self.assertEqual(self.n14.outputs.get_shape().as_list()[1:], [6, 6, 32])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
