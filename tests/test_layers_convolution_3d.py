#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))

        cls.input_layer = tl.layers.Input(name='input_layer')(x)

        cls.n1 = tl.layers.Conv3dLayer(shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))(cls.input_layer)

        cls.n2 = tl.layers.DeConv3dLayer(shape=(2, 2, 2, 128, 32), strides=(1, 2, 2, 2, 1))(cls.n1)

        cls.n3 = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))(cls.n2)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_weights), 2)
        self.assertEqual(self.n1.count_weights(), 800)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 50, 50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_weights), 4)
        self.assertEqual(self.n2.count_weights(), 33696)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [99, 99, 99, 128])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_weights), 6)
        self.assertEqual(self.n3.count_weights(), 144320)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [199, 199, 199, 32])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
