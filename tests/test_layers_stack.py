#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Stack_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, shape=[None, 30])
        net_in = tl.layers.InputLayer(x, name='input')

        net = tl.layers.DropoutLayer(net_in, keep=0.5, is_train=True, name='dropout')

        net_d1 = tl.layers.DenseLayer(net, n_units=10, name='dense1')
        net_d2 = tl.layers.DenseLayer(net, n_units=10, name='dense2')
        net_d3 = tl.layers.DenseLayer(net, n_units=10, name='dense3')

        cls.net_stack = tl.layers.StackLayer([net_d1, net_d2, net_d3], axis=1, name='stack')

        cls.net_stack.print_layers()
        cls.net_stack.print_params(False)

        cls.net_unstack = tl.layers.UnStackLayer(cls.net_stack, axis=1, name='unstack')

        cls.net_unstack.print_layers()
        cls.net_unstack.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_StackLayer(self):
        self.assertEqual(self.net_stack.outputs.get_shape().as_list()[-1], 10)
        self.assertEqual(len(self.net_stack.all_layers), 6)
        self.assertEqual(len(self.net_stack.all_weights), 6)
        self.assertEqual(len(self.net_stack.all_drop), 1)
        self.assertEqual(self.net_stack.count_weights(), 930)

    def test_UnStackLayer(self):

        for n in self.net_unstack.outputs:
            shape = n.outputs.get_shape().as_list()

            self.assertEqual(shape[-1], 10)
            self.assertEqual(len(n.all_layers), 7)
            self.assertEqual(len(n.all_weights), 6)
            self.assertEqual(len(n.all_drop), 1)
            self.assertEqual(n.count_weights(), 930)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
