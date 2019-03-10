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

        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.DenseLayer(n, n_units=80, name='d1')
        n = tl.layers.DenseLayer(n, n_units=80, name='d2')

        n.print_layers()
        n.print_params(False)

        n2 = n[:, :30]
        n2.print_layers()

        cls.n_params = n.count_params()
        cls.all_layers = n.all_layers
        cls.all_params = n.all_params
        cls.shape_n = n.outputs.get_shape().as_list()

        cls.shape_n2 = n2.outputs.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_n_params(self):
        self.assertEqual(self.n_params, 14560)

    def test_shape_n(self):
        self.assertEqual(self.shape_n[-1], 80)

    def test_all_layers(self):
        self.assertEqual(len(self.all_layers), 3)

    def test_all_params(self):
        self.assertEqual(len(self.all_params), 4)

    def test_shape_n2(self):
        self.assertEqual(self.shape_n2[-1], 30)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
