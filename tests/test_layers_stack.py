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
        net_in = tl.layers.Input(name='input')(x)

        net = tl.layers.Dropout(keep=0.5, name='dropout')(net_in, is_train=True)

        net_d1 = tl.layers.Dense(n_units=10, name='dense1')(net)
        net_d2 = tl.layers.Dense(n_units=10, name='dense2')(net)
        net_d3 = tl.layers.Dense(n_units=10, name='dense3')(net)

        # print(net_d3.outputs)
        # print(net_d3.all_layers)
        # print(net_d3.all_weights)
        # exit()

        cls.net_stack = tl.layers.Stack(axis=1, name='stack')([net_d1, net_d2, net_d3])

        # cls.net_stack.print_layers()
        # cls.net_stack.print_weights(False)

        cls.net_unstack = tl.layers.UnStack(axis=1, name='unstack')(cls.net_stack)

        # cls.net_unstack.print_layers()
        # cls.net_unstack.print_weights(False)
        exit()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_Stack(self):
        self.assertEqual(self.net_stack.outputs.get_shape().as_list()[-1], 10)
        self.assertEqual(len(self.net_stack.all_layers), 6)
        self.assertEqual(len(self.net_stack.all_weights), 6)
        self.assertEqual(len(self.net_stack.all_drop), 1)
        self.assertEqual(self.net_stack.count_weights(), 930)

    def test_UnStack(self):

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
