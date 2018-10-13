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
        n = tl.layers.Input(name='in')(t_signal)
        n = tl.layers.Conv1d(n_filter=32, filter_size=3, stride=1, padding='SAME', name='conv1d')(n)
        net1 = tl.layers.SubpixelConv1d(scale=2, name='subpixel')(n)

        net1.print_layers()
        net1.print_weights(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_weights = net1.all_weights
        cls.net1_n_weights = net1.count_weights()

        ## 2D
        x = tf.placeholder('float32', [10, 100, 100, 3], name='x')
        n = tl.layers.Input(name='in')(x)
        n = tl.layers.Conv2d(n_filter=32, filter_size=(3, 2), strides=(1, 1), padding='SAME', name='conv2d')(n)
        net2 = tl.layers.SubpixelConv2d(scale=2, name='subpixel2d')(n)

        net2.print_layers()
        net2.print_weights(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_weights = net2.all_weights
        cls.net2_n_weights = net2.count_weights()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1_shape(self):
        self.assertEqual(self.net1_shape, [10, 200, 16])
        self.assertEqual(len(self.net1_layers), 3)
        self.assertEqual(len(self.net1_weights), 2)
        self.assertEqual(self.net1_n_weights, 416)

    def test_net2_shape(self):
        self.assertEqual(self.net2_shape, [10, 200, 200, 8])
        self.assertEqual(len(self.net2_layers), 3)
        self.assertEqual(len(self.net2_weights), 2)
        self.assertEqual(self.net2_n_weights, 608)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
