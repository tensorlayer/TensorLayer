#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl


class Layer_Convolution_3D_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))

        cls.input_layer = tl.layers.InputLayer(x, name='input_layer')

        cls.n1 = tl.layers.Conv3dLayer(cls.input_layer, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))

        cls.n2 = tl.layers.DeConv3dLayer(
            cls.n1, shape=(2, 2, 2, 128, 32), output_shape=(100, 12, 32, 32, 128), strides=(1, 2, 2, 2, 1)
        )

        cls.n3 = tl.layers.DeConv3d(cls.n2, n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.n1.all_layers), 2)
        self.assertEqual(len(self.n1.all_params), 2)
        self.assertEqual(self.n1.count_params(), 800)
        self.assertEqual(self.n1.outputs.get_shape().as_list()[1:], [50, 50, 50, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.n2.all_layers), 3)
        self.assertEqual(len(self.n2.all_params), 4)
        self.assertEqual(self.n2.count_params(), 33696)
        self.assertEqual(self.n2.outputs.get_shape().as_list()[1:], [12, 32, 32, 128])

    def test_layer_n3(self):

        self.assertEqual(len(self.n3.all_layers), 4)
        self.assertEqual(len(self.n3.all_params), 6)
        self.assertEqual(self.n3.count_params(), 144320)
        self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [24, 64, 64, 32])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
