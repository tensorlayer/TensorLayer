#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Padding_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        ## 1D
        x = tf.placeholder(tf.float32, (None, 100, 1))
        n = tl.layers.InputLayer(x)

        n1 = tl.layers.ZeroPad1d(n, padding=1)
        n2 = tl.layers.ZeroPad1d(n, padding=(2, 3))

        n1.print_layers()
        n2.print_layers()

        cls.n1_shape = n1.outputs.get_shape().as_list()
        cls.n2_shape = n2.outputs.get_shape().as_list()

        ## 2D
        x = tf.placeholder(tf.float32, (None, 100, 100, 3))
        n = tl.layers.InputLayer(x)

        n3 = tl.layers.ZeroPad2d(n, padding=2)
        n4 = tl.layers.ZeroPad2d(n, padding=(2, 3))
        n5 = tl.layers.ZeroPad2d(n, padding=((3, 3), (4, 4)))

        n3.print_layers()
        n4.print_layers()
        n5.print_layers()

        cls.n3_shape = n3.outputs.get_shape().as_list()
        cls.n4_shape = n4.outputs.get_shape().as_list()
        cls.n5_shape = n5.outputs.get_shape().as_list()

        ## 3D
        x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
        n = tl.layers.InputLayer(x)

        n6 = tl.layers.ZeroPad3d(n, padding=2)
        n7 = tl.layers.ZeroPad3d(n, padding=(2, 3, 4))
        n8 = tl.layers.ZeroPad3d(n, padding=((3, 3), (4, 4), (5, 5)))

        n6.print_layers()
        n7.print_layers()
        n8.print_layers()

        cls.n6_shape = n6.outputs.get_shape().as_list()
        cls.n7_shape = n7.outputs.get_shape().as_list()
        cls.n8_shape = n8.outputs.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_n1_shape(self):
        self.assertEqual(self.n1_shape[1:], [102, 1])

    def test_n2_shape(self):
        self.assertEqual(self.n2_shape[1:], [105, 1])

    def test_n3_shape(self):
        self.assertEqual(self.n3_shape[1:], [104, 104, 3])

    def test_n4_shape(self):
        self.assertEqual(self.n4_shape[1:], [104, 106, 3])

    def test_n5_shape(self):
        self.assertEqual(self.n5_shape[1:], [106, 108, 3])

    def test_n6_shape(self):
        self.assertEqual(self.n6_shape[1:], [104, 104, 104, 3])

    def test_n7_shape(self):
        self.assertEqual(self.n7_shape[1:], [104, 106, 108, 3])

    def test_n8_shape(self):
        self.assertEqual(self.n8_shape[1:], [106, 108, 110, 3])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
