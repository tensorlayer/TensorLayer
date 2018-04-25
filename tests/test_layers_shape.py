#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Shape_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        net = tl.layers.InputLayer(x, name='input')

        ## Flatten
        net1 = tl.layers.FlattenLayer(net, name='flatten')

        net1.print_layers()
        net1.print_params(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_params = net1.all_params
        cls.net1_n_params = net1.count_params()

        ## Reshape
        net2 = tl.layers.ReshapeLayer(net1, shape=(-1, 28, 28, 1), name='reshape')

        net2.print_layers()
        net2.print_params(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_params = net2.all_params
        cls.net2_n_params = net2.count_params()

        ## TransposeLayer
        net3 = tl.layers.TransposeLayer(net2, perm=[0, 1, 3, 2], name='trans')

        net3.print_layers()
        net3.print_params(False)

        cls.net3_shape = net3.outputs.get_shape().as_list()
        cls.net3_layers = net3.all_layers
        cls.net3_params = net3.all_params
        cls.net3_n_params = net3.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1_shape(self):
        self.assertEqual(self.net1_shape[-1], 784)

    def test_net1_layers(self):
        self.assertEqual(len(self.net1_layers), 1)

    def test_net1_params(self):
        self.assertEqual(len(self.net1_params), 0)

    def test_net1_n_params(self):
        self.assertEqual(self.net1_n_params, 0)

    def test_net2_shape(self):
        self.assertEqual(self.net2_shape[1:], [28, 28, 1])

    def test_net2_layers(self):
        self.assertEqual(len(self.net2_layers), 2)

    def test_net2_params(self):
        self.assertEqual(len(self.net2_params), 0)

    def test_net2_n_params(self):
        self.assertEqual(self.net2_n_params, 0)

    def test_net3_shape(self):
        self.assertEqual(self.net3_shape[1:], [28, 1, 28])

    def test_net3_layers(self):
        self.assertEqual(len(self.net3_layers), 3)

    def test_net3_params(self):
        self.assertEqual(len(self.net3_params), 0)

    def test_net3_n_params(self):
        self.assertEqual(self.net3_n_params, 0)


if __name__ == '__main__':

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
