#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class PReLU_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.InputLayer(x, name='input')

        net = tl.layers.DenseLayer(in_layer, n_units=10, name='dense_1')
        cls.net1 = tl.layers.PReluLayer(net, name='prelu_1')

        cls.net1.print_layers()
        cls.net1.print_params(False)

        net2 = tl.layers.DenseLayer(cls.net1, n_units=30, name='dense_2')
        cls.net2 = tl.layers.PReluLayer(net2, channel_shared=True, name='prelu_2')

        cls.net2.print_layers()
        cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(len(self.net1.all_layers), 3)
        self.assertEqual(len(self.net1.all_params), 3)
        self.assertEqual(self.net1.count_params(), 320)
        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        self.assertEqual(len(self.net2.all_layers), 5)
        self.assertEqual(len(self.net2.all_params), 6)
        self.assertEqual(self.net2.count_params(), 651)
        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


class PRelu6_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.InputLayer(x, name='input')

        net = tl.layers.DenseLayer(in_layer, n_units=10, name='dense_1')
        cls.net1 = tl.layers.PRelu6Layer(net, name='prelu6_1')

        cls.net1.print_layers()
        cls.net1.print_params(False)

        net2 = tl.layers.DenseLayer(cls.net1, n_units=30, name='dense_2')
        cls.net2 = tl.layers.PRelu6Layer(net2, channel_shared=True, name='prelu6_2')

        cls.net2.print_layers()
        cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(len(self.net1.all_layers), 3)
        self.assertEqual(len(self.net1.all_params), 3)
        self.assertEqual(self.net1.count_params(), 320)
        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        self.assertEqual(len(self.net2.all_layers), 5)
        self.assertEqual(len(self.net2.all_params), 6)
        self.assertEqual(self.net2.count_params(), 651)
        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


class PTRelu6_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])

        in_layer = tl.layers.InputLayer(x, name='input')

        net = tl.layers.DenseLayer(in_layer, n_units=10, name='dense_1')
        cls.net1 = tl.layers.PTRelu6Layer(net, name='ptrelu6_1')

        cls.net1.print_layers()
        cls.net1.print_params(False)

        net2 = tl.layers.DenseLayer(cls.net1, n_units=30, name='dense_2')
        cls.net2 = tl.layers.PTRelu6Layer(net2, channel_shared=True, name='ptrelu6_2')

        cls.net2.print_layers()
        cls.net2.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(len(self.net1.all_layers), 3)
        self.assertEqual(len(self.net1.all_params), 4)
        self.assertEqual(self.net1.count_params(), 330)
        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [10])

        prelu1_param_shape = self.net1.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu1_param_shape, [10])

    def test_net2(self):
        self.assertEqual(len(self.net2.all_layers), 5)
        self.assertEqual(len(self.net2.all_params), 8)
        self.assertEqual(self.net2.count_params(), 662)
        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [30])

        prelu2_param_shape = self.net2.all_params[-1].get_shape().as_list()
        self.assertEqual(prelu2_param_shape, [1])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
