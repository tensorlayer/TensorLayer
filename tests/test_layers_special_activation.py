#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Special_Activation_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DenseLayer(net, n_units=10, name='dense')
        net1 = tl.layers.PReluLayer(net, name='prelu')

        net1.print_layers()
        net1.print_params(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_params = net1.all_params
        cls.net1_n_params = net1.count_params()

        net2 = tl.layers.PReluLayer(net1, channel_shared=True, name='prelu2')

        net2.print_layers()
        net2.print_params(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_params = net2.all_params
        cls.net2_n_params = net2.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(len(self.net1_layers), 3)
        self.assertEqual(len(self.net1_params), 3)
        self.assertEqual(self.net1_n_params, 320)
        self.assertEqual(self.net1_shape[-1], 10)

    def test_net2(self):
        self.assertEqual(len(self.net2_layers), 4)
        self.assertEqual(len(self.net2_params), 4)
        self.assertEqual(self.net2_n_params, 321)
        self.assertEqual(self.net2_shape[-1], 10)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
