#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Flow_Control_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        x = tf.placeholder(tf.float32, shape=(None, 784), name='x')

        # define the network
        net_in = tl.layers.InputLayer(x, name='in')
        net_in = tl.layers.DropoutLayer(net_in, keep=0.8, name='in/drop')
        # net 0
        net_0 = tl.layers.DenseLayer(net_in, n_units=800, act=tf.nn.relu, name='net0/relu1')
        net_0 = tl.layers.DropoutLayer(net_0, keep=0.5, name='net0/drop1')
        net_0 = tl.layers.DenseLayer(net_0, n_units=800, act=tf.nn.relu, name='net0/relu2')
        # net 1
        net_1 = tl.layers.DenseLayer(net_in, n_units=800, act=tf.nn.relu, name='net1/relu1')
        net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop1')
        net_1 = tl.layers.DenseLayer(net_1, n_units=800, act=tf.nn.relu, name='net1/relu2')
        net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop2')
        net_1 = tl.layers.DenseLayer(net_1, n_units=800, act=tf.nn.relu, name='net1/relu3')
        # multiplexer
        net_mux = tl.layers.MultiplexerLayer(layers=[net_0, net_1], name='mux')
        network = tl.layers.ReshapeLayer(net_mux, shape=(-1, 800), name='reshape')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
        # output layer
        network = tl.layers.DenseLayer(network, n_units=10, name='output')

        network.print_layers()
        network.print_params(False)

        cls.net_shape = network.outputs.get_shape().as_list()
        cls.net_layers = network.all_layers
        cls.net_params = network.all_params
        cls.net_all_drop = network.all_drop
        cls.net_n_params = network.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net_shape(self):
        self.assertEqual(self.net_shape[-1], 10)

    def test_net_layers(self):
        self.assertEqual(len(self.net_layers), 14)

    def test_net_params(self):
        self.assertEqual(len(self.net_params), 12)

    def test_net_all_drop(self):
        self.assertEqual(len(self.net_all_drop), 5)

    def test_net_n_params(self):
        self.assertEqual(self.net_n_params, 3186410)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
