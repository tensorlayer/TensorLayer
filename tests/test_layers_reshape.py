#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Shape_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        net = tl.layers.Input(name='input')(x)

        ## Flatten
        net1 = tl.layers.Flatten(name='flatten')(net)

        net1.print_layers()
        net1.print_weights(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_weights = net1.all_weights
        cls.net1_n_weights = net1.count_weights()

        ## Reshape
        net2 = tl.layers.Reshape(shape=(-1, 28, 28, 1), name='reshape')(net1)

        net2.print_layers()
        net2.print_weights(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_weights = net2.all_weights
        cls.net2_n_weights = net2.count_weights()

        ## TransposeLayer
        net3 = tl.layers.Transpose(perm=[0, 1, 3, 2], name='trans')(net2)

        net3.print_layers()
        net3.print_weights(False)

        cls.net3_shape = net3.outputs.get_shape().as_list()
        cls.net3_layers = net3.all_layers
        cls.net3_weights = net3.all_weights
        cls.net3_n_weights = net3.count_weights()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(len(self.net1_layers), 2)
        self.assertEqual(self.net1_shape[-1], 784)
        self.assertEqual(len(self.net1_weights), 0)
        self.assertEqual(self.net1_n_weights, 0)

    def test_net2(self):
        self.assertEqual(len(self.net2_layers), 3)
        self.assertEqual(self.net2_shape[1:], [28, 28, 1])
        self.assertEqual(len(self.net2_weights), 0)
        self.assertEqual(self.net2_n_weights, 0)

    def test_net3(self):
        self.assertEqual(len(self.net3_layers), 4)
        self.assertEqual(self.net3_shape[1:], [28, 1, 28])
        self.assertEqual(len(self.net3_weights), 0)
        self.assertEqual(self.net3_n_weights, 0)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
