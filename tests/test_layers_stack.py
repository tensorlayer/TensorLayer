#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Stack_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])
        net = tl.layers.InputLayer(x, name='input')
        net1 = tl.layers.DenseLayer(net, n_units=10, name='dense1')
        net2 = tl.layers.DenseLayer(net, n_units=10, name='dense2')
        net3 = tl.layers.DenseLayer(net, n_units=10, name='dense3')
        net = tl.layers.StackLayer([net1, net2, net3], axis=1, name='stack')

        net.print_layers()
        net.print_params(False)

        cls.net_shape = net.outputs.get_shape().as_list()
        cls.layers = net.all_layers
        cls.params = net.all_params
        cls.n_params = net.count_params()

        cls.net = tl.layers.UnStackLayer(net, axis=1, name='unstack')

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net_shape(self):
        self.assertEqual(self.net_shape[-1], 10)

    def test_layers(self):
        self.assertEqual(len(self.layers), 4)

    def test_params(self):
        self.assertEqual(len(self.params), 6)
        self.assertEqual(self.n_params, 930)

    def test_unstack(self):

        for n in self.net:
            shape = n.outputs.get_shape().as_list()

            self.assertEqual(shape[-1], 10)
            self.assertEqual(len(n.all_layers), 4)
            self.assertEqual(len(n.all_params), 6)
            self.assertEqual(n.count_params(), 930)


if __name__ == '__main__':

    # tl.logging.set_verbosity(tl.logging.INFO)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
