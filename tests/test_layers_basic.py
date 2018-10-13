#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Basic_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, [None, 100])

        n = tl.layers.Input(name='in')(x)
        n = tl.layers.Dense(n_units=80, name='d1')(n)
        n = tl.layers.Dense(n_units=80, name='d2')(n)

        n.print_layers()
        n.print_weights(False)

        n2 = n[:, :30]
        n2.print_layers()

        cls.n_weights = n.count_weights()
        cls.all_layers = n.all_layers
        cls.all_weights = n.all_weights
        cls.shape_n = n.outputs.get_shape().as_list()

        cls.shape_n2 = n2.outputs.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_n_weights(self):
        self.assertEqual(self.n_weights, 14560)

    def test_shape_n(self):
        self.assertEqual(self.shape_n[-1], 80)

    def test_all_layers(self):
        self.assertEqual(len(self.all_layers), 3)

    def test_all_weights(self):
        self.assertEqual(len(self.all_weights), 4)

    def test_shape_n2(self):
        self.assertEqual(self.shape_n2[-1], 30)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
