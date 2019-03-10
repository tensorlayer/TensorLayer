#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Super_Resolution_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        t_signal = tf.placeholder('float32', [10, 100, 4], name='x')
        n = tl.layers.InputLayer(t_signal, name='in')
        n = tl.layers.Conv1d(n, n_filter=32, filter_size=3, stride=1, padding='SAME', name='conv1d')
        net1 = tl.layers.SubpixelConv1d(n, scale=2, name='subpixel')

        net1.print_layers()
        net1.print_params(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_params = net1.all_params
        cls.net1_n_params = net1.count_params()

        ## 2D
        x = tf.placeholder('float32', [10, 100, 100, 3], name='x')
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.Conv2d(n, n_filter=32, filter_size=(3, 2), strides=(1, 1), padding='SAME', name='conv2d')
        net2 = tl.layers.SubpixelConv2d(n, scale=2, name='subpixel2d')

        net2.print_layers()
        net2.print_params(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_params = net2.all_params
        cls.net2_n_params = net2.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1_shape(self):
        self.assertEqual(self.net1_shape, [10, 200, 16])
        self.assertEqual(len(self.net1_layers), 3)
        self.assertEqual(len(self.net1_params), 2)
        self.assertEqual(self.net1_n_params, 416)

    def test_net2_shape(self):
        self.assertEqual(self.net2_shape, [10, 200, 200, 8])
        self.assertEqual(len(self.net2_layers), 3)
        self.assertEqual(len(self.net2_params), 2)
        self.assertEqual(self.net2_n_params, 608)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
