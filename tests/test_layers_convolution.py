#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 1))

        cls.input_layer = tl.layers.InputLayer(x, name='input_layer')

        print("input:", cls.input_layer.all_layers)

        cls.n1 = tl.layers.Conv1dLayer(cls.input_layer, shape=(5, 1, 32), stride=2)

        cls.n2 = tl.layers.Conv1d(cls.n1, n_filter=32, filter_size=5, stride=2)

        cls.n3 = tl.layers.SeparableConv1d(
            cls.n2, n_filter=32, filter_size=3, strides=1, padding='VALID', act=tf.nn.relu, name='separable_1d'
        )

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_params), 2)
        self.assertEqual(self.n1.count_params(), 192)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_params), 4)
        self.assertEqual(self.n2.count_params(), 5344)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [25, 32])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_params), 7)
        self.assertEqual(self.n3.count_params(), 6496)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [23, 32])


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 100, 3))

        cls.input_layer = tl.layers.InputLayer(x, name='input_layer')

        print(cls.input_layer.all_layers)

        cls.n1 = tl.layers.Conv2dLayer(
            cls.input_layer, act=tf.nn.relu, shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init=tf.constant_initializer(value=0.0),
            name='conv2dlayer'
        )

        print("#################################")
        print("input:", cls.input_layer.all_layers)
        print("input:", cls.n1.all_layers)

        cls.n2 = tl.layers.Conv2d(cls.n1, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d')

        cls.n3 = tl.layers.Conv2d(
            cls.n2, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, b_init=None, name='conv2d_no_bias'
        )

        cls.n4 = tl.layers.DeConv2dLayer(
            cls.n3, shape=(5, 5, 32, 32), output_shape=(100, 200, 200, 32), strides=(1, 2, 2, 1), name='deconv2dlayer'
        )

        cls.n5 = tl.layers.DeConv2d(cls.n4, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d')

        cls.n6 = tl.layers.DepthwiseConv2d(
            cls.n5, shape=(3, 3), strides=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise'
        )

        cls.n7 = tl.layers.Conv2d(
            cls.n6, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='conv2d2'
        )

        cls.n8 = tl.layers.GroupConv2d(cls.n7, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='group')

        cls.n9 = tl.layers.SeparableConv2d(
            cls.n8, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, name='seperable2d1'
        )

        cls.n10 = tl.layers.TernaryConv2d(cls.n9, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')

        cls.n11 = tl.layers.AtrousDeConv2dLayer(
            cls.n10, shape=(3, 3, 32, 64), output_shape=(100, 96, 96, 32), rate=2, act=tf.nn.relu, name='atroustrans1'
        )

        cls.n12 = tl.layers.QuanConv2d(cls.n11, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='quancnn')

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_params), 2)
        self.assertEqual(self.n1.count_params(), 2432)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_params), 4)
        self.assertEqual(self.n2.count_params(), 11680)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [25, 25, 32])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_params), 5)
        self.assertEqual(self.n3.count_params(), 20896)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [13, 13, 32])

    def test_layer_n4(self):

        self.assertEqual(len(self.n4.all_layers), 5)
        self.assertEqual(len(self.n4.all_params), 7)
        self.assertEqual(self.n4.count_params(), 46528)
        self.assertEqual(self.n4.outputs.get_shape().as_list()[1:], [200, 200, 32])

    def test_layer_n5(self):

        self.assertEqual(len(self.n5.all_layers), 6)
        self.assertEqual(len(self.n5.all_params), 9)
        self.assertEqual(self.n5.count_params(), 55776)
        self.assertEqual(self.n5.outputs.get_shape().as_list()[1:], [400, 400, 32])

    def test_layer_n6(self):

        self.assertEqual(len(self.n6.all_layers), 7)
        self.assertEqual(len(self.n6.all_params), 11)
        self.assertEqual(self.n6.count_params(), 56416)
        self.assertEqual(self.n6.outputs.get_shape().as_list()[1:], [200, 200, 64])

    def test_layer_n7(self):

        self.assertEqual(len(self.n7.all_layers), 8)
        self.assertEqual(len(self.n7.all_params), 13)
        self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(self.n7.outputs.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n8(self):

        self.assertEqual(len(self.n8.all_layers), 9)
        self.assertEqual(len(self.n8.all_params), 15)
        self.assertEqual(self.n8.count_params(), 79520)
        self.assertEqual(self.n8.outputs.get_shape().as_list()[1:], [50, 50, 32])

    def test_layer_n9(self):

        self.assertEqual(len(self.n9.all_layers), 10)
        self.assertEqual(len(self.n9.all_params), 18)
        self.assertEqual(self.n9.count_params(), 80864)
        self.assertEqual(self.n9.outputs.get_shape().as_list()[1:], [48, 48, 32])

    def test_layer_n10(self):

        self.assertEqual(len(self.n10.all_layers), 11)
        self.assertEqual(len(self.n10.all_params), 20)
        self.assertEqual(self.n10.count_params(), 132128)
        self.assertEqual(self.n10.outputs.get_shape().as_list()[1:], [48, 48, 64])

    def test_layer_n11(self):

        self.assertEqual(len(self.n11.all_layers), 12)
        self.assertEqual(len(self.n11.all_params), 22)
        self.assertEqual(self.n11.count_params(), 150592)
        self.assertEqual(self.n11.outputs.get_shape().as_list()[1:], [96, 96, 32])

    def test_layer_n12(self):

        self.assertEqual(len(self.n12.all_layers), 13)
        self.assertEqual(len(self.n12.all_params), 24)
        self.assertEqual(self.n12.count_params(), 201856)
        self.assertEqual(self.n12.outputs.get_shape().as_list()[1:], [96, 96, 64])


class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))

        cls.input_layer = tl.layers.InputLayer(x, name='input_layer')

        cls.n1 = tl.layers.Conv3dLayer(cls.input_layer, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))

        cls.n2 = tl.layers.DeConv3dLayer(
            cls.n1, shape=(2, 2, 2, 128, 32), output_shape=(100, 12, 32, 32, 128), strides=(1, 2, 2, 2, 1)
        )

        cls.n3 = tl.layers.DeConv3d(cls.n2, n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_params), 2)
        self.assertEqual(self.n1.count_params(), 800)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 50, 50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_params), 4)
        self.assertEqual(self.n2.count_params(), 33696)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [12, 32, 32, 128])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_params), 6)
        self.assertEqual(self.n3.count_params(), 144320)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [24, 64, 64, 32])


class Layer_DeformableConvolution_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, [None, 299, 299, 3])
        net = tl.layers.InputLayer(x, name='input_layer')

        print("input:", net.all_layers)

        offset1 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset1')
        cls.net1 = tl.layers.DeformableConv2d(net, offset1, 32, (3, 3), act=tf.nn.relu, name='deformable1')

        offset2 = tl.layers.Conv2d(cls.net1, 18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset2')
        cls.net2 = tl.layers.DeformableConv2d(cls.net1, offset2, 64, (3, 3), act=tf.nn.relu, name='deformable2')

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.net1.all_layers), 2)
        self.assertEqual(len(self.net1.all_params), 2)
        self.assertEqual(self.net1.count_params(), 896)
        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [299, 299, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.net2.all_layers), 3)
        self.assertEqual(len(self.net2.all_params), 4)
        self.assertEqual(self.net2.count_params(), 19392)
        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [299, 299, 64])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
