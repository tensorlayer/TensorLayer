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

        def get_network_1d(inputs, reuse=False):

            with tf.variable_scope("1D_network", reuse=reuse):
                net = tl.layers.Input()(network)

                net1 = tl.layers.Conv1d(name="Conv1d")(net)  # 2 params
                net2 = tl.layers.SeparableConv1d(name="SeparableConv1d")(net1)  # 3 params
                net3 = tl.layers.MaxPool1d((1, ), name="MaxPool1d")(net2)  # 0 params
                net4 = tl.layers.MeanPool1d((1, ), name="MeanPool1d")(net3)  # 0 params

                # HAO Test
                net5 = tl.layers.Conv1d(name="Conv1d1")(net4)  # 2 params
                net6 = tl.layers.SeparableConv1d(name="SeparableConv1d1")(net5)  # 3 params
                net7 = tl.layers.SeparableConv1d(name="SeparableConv1d2")(net6)  # 3 params

            return [net, net1, net2, net3, net4, net5, net6, net7]

        input_pl_train = tf.placeholder(tf.float32, [None, 32, 3])
        input_plh_test = tf.placeholder(tf.float32, [None, 32, 3])

        cls.network_1 = get_network_1d(input_pl_train, reuse=False)
        cls.network_2 = get_network_1d(input_plh_test, reuse=True)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_net0(self):
        self.assertEqual(len(self.network_1[0].all_weights), 0)
        self.assertEqual(len(self.network_2[0].all_weights), 0)

    def test_layer_net1(self):
        self.assertEqual(len(self.network_1[1].all_weights), 2)
        self.assertEqual(len(self.network_2[1].all_weights), 2)

    def test_layer_net2(self):
        self.assertEqual(len(self.network_1[2].all_weights), 5)
        self.assertEqual(len(self.network_2[2].all_weights), 5)

    def test_layer_net3(self):
        self.assertEqual(len(self.network_1[3].all_weights), 5)
        self.assertEqual(len(self.network_2[3].all_weights), 5)

    def test_layer_net4(self):
        self.assertEqual(len(self.network_1[4].all_weights), 5)
        self.assertEqual(len(self.network_2[4].all_weights), 5)

    def test_layer_net5(self):
        self.assertEqual(len(self.network_1[5].all_weights), 7)
        self.assertEqual(len(self.network_2[5].all_weights), 7)

    def test_layer_net6(self):
        self.assertEqual(len(self.network_1[6].all_weights), 10)
        self.assertEqual(len(self.network_2[6].all_weights), 10)

    def test_layer_net7(self):
        self.assertEqual(len(self.network_1[7].all_weights), 13)
        self.assertEqual(len(self.network_2[7].all_weights), 13)


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        def get_network_2d(inputs, reuse=False):

            with tf.variable_scope("2D_network", reuse=reuse):
                net = tl.layers.Input()(inputs)

                net1 = tl.layers.Conv2d(name="Conv2d")(net)  # 2 params
                net2 = tl.layers.DeConv2d(name="DeConv2d")(net1)  # 2 params
                net3 = tl.layers.SeparableConv2d(name="SeparableConv2d")(net2)  # 3 params
                net4 = tl.layers.MaxPool2d((1, 1), name="MaxPool2d")(net3)  # 0 params
                net5 = tl.layers.MeanPool2d((1, 1), name="MeanPool2d")(net4)  # 0 params

                # HAO Test
                net6 = tl.layers.Conv2d(name="Conv2d1")(net5)  # 2 params
                net7 = tl.layers.DeConv2d(name="DeConv2d1")(net6)  # 2 params
                net8 = tl.layers.DeConv2d(name="DeConv2d2")(net7)  # 2 params
                net9 = tl.layers.SeparableConv2d(name="SeparableConv2d1")(net8)  # 3 params

            return [net, net1, net2, net3, net4, net5, net6, net7, net8, net9]

        input_pl_train = tf.placeholder(tf.float32, [None, 32, 32, 3])
        input_plh_test = tf.placeholder(tf.float32, [None, 32, 32, 3])

        cls.network_1 = get_network_2d(input_pl_train, reuse=False)
        cls.network_2 = get_network_2d(input_plh_test, reuse=True)

    def test_layer_net0(self):
        self.assertEqual(len(self.network_1[0].all_weights), 0)
        self.assertEqual(len(self.network_2[0].all_weights), 0)

    def test_layer_net1(self):
        self.assertEqual(len(self.network_1[1].all_weights), 2)
        self.assertEqual(len(self.network_2[1].all_weights), 2)

    def test_layer_net2(self):
        self.assertEqual(len(self.network_1[2].all_weights), 4)
        self.assertEqual(len(self.network_2[2].all_weights), 4)

    def test_layer_net3(self):
        self.assertEqual(len(self.network_1[3].all_weights), 7)
        self.assertEqual(len(self.network_2[3].all_weights), 7)

    def test_layer_net4(self):
        self.assertEqual(len(self.network_1[4].all_weights), 7)
        self.assertEqual(len(self.network_2[4].all_weights), 7)

    def test_layer_net5(self):
        self.assertEqual(len(self.network_1[5].all_weights), 7)
        self.assertEqual(len(self.network_2[5].all_weights), 7)

    def test_layer_net6(self):
        self.assertEqual(len(self.network_1[6].all_weights), 9)
        self.assertEqual(len(self.network_2[6].all_weights), 9)

    def test_layer_net7(self):
        self.assertEqual(len(self.network_1[7].all_weights), 11)
        self.assertEqual(len(self.network_2[7].all_weights), 11)

    def test_layer_net8(self):
        self.assertEqual(len(self.network_1[8].all_weights), 13)
        self.assertEqual(len(self.network_2[8].all_weights), 13)

    def test_layer_net9(self):
        self.assertEqual(len(self.network_1[9].all_weights), 16)
        self.assertEqual(len(self.network_2[9].all_weights), 16)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()


class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        def get_network_3d(inputs, reuse=False):

            with tf.variable_scope("3D_network", reuse=reuse):
                net = tl.layers.Input()(inputs)

                net1 = tl.layers.Conv3dLayer(
                    shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1), name="Conv3dLayer"
                )(net)  # 2 params
                net2 = tl.layers.DeConv3d(name="DeConv3d")(net1)  # 2 params
                net3 = tl.layers.MaxPool3d((1, 1, 1), name="MaxPool3d")(net2)  # 0 params
                net4 = tl.layers.MeanPool3d((1, 1, 1), name="MeanPool3d")(net3)  # 0 params

                # HAO Test
                net5 = tl.layers.DeConv3d(name="DeConv3d1")(net4)  # 2 params

                return [net, net1, net2, net3, net4, net5]

        input_pl_train = tf.placeholder(tf.float32, [None, 32, 32, 32, 3])
        input_plh_test = tf.placeholder(tf.float32, [None, 32, 32, 32, 3])

        cls.network_1 = get_network_3d(input_pl_train, reuse=False)
        cls.network_2 = get_network_3d(input_plh_test, reuse=True)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_net0(self):
        self.assertEqual(len(self.network_1[0].all_weights), 0)
        self.assertEqual(len(self.network_2[0].all_weights), 0)

    def test_layer_net1(self):
        self.assertEqual(len(self.network_1[1].all_weights), 2)
        self.assertEqual(len(self.network_2[1].all_weights), 2)

    def test_layer_net2(self):
        self.assertEqual(len(self.network_1[2].all_weights), 4)
        self.assertEqual(len(self.network_2[2].all_weights), 4)

    def test_layer_net3(self):
        self.assertEqual(len(self.network_1[3].all_weights), 4)
        self.assertEqual(len(self.network_2[3].all_weights), 4)

    def test_layer_net4(self):
        self.assertEqual(len(self.network_1[4].all_weights), 4)
        self.assertEqual(len(self.network_2[4].all_weights), 4)

    def test_layer_net5(self):
        self.assertEqual(len(self.network_1[5].all_weights), 6)
        self.assertEqual(len(self.network_2[5].all_weights), 6)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
