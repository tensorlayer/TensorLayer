#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Basic_Test(unittest.TestCase):
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

    def test_net1_shape(self):
        assert (self.net1_shape[-1] == 10)

    def test_net1_all_layers(self):
        assert (len(self.net1_layers) == 2)

    def test_net1_all_params(self):
        assert (len(self.net1_params) == 3)

    def test_net1_n_params(self):
        assert (self.net1_n_params == 320)

    def test_net2_shape(self):
        assert (self.net2_shape[-1] == 10)

    def test_net2_all_layers(self):
        assert (len(self.net2_layers) == 3)

    def test_net2_all_params(self):
        assert (len(self.net2_params) == 4)

    def test_net2_n_params(self):
        assert (self.net2_n_params == 321)


if __name__ == '__main__':

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
