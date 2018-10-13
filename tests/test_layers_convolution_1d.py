#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 1))

        cls.input_layer = tl.layers.Input(name='input_layer')(x)

        cls.n1 = tl.layers.Conv1d(shape=(5, 1, 32), stride=2)(cls.input_layer)

        cls.n2 = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2, padding="same")(cls.n1)

        cls.n3 = tl.layers.SeparableConv1d(
            n_filter=32, filter_size=3, strides=1, padding='VALID', act=tf.nn.relu, name='separable_1d'
        )(cls.n2)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_weights), 2)
        self.assertEqual(self.n1.count_weights(), 192)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_weights), 4)
        self.assertEqual(self.n2.count_weights(), 5344)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [25, 32])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_weights), 7)
        self.assertEqual(self.n3.count_weights(), 6496)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [23, 32])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
