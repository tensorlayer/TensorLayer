#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl


class Layer_DeformableConvolution_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, [None, 299, 299, 3])
        net = tl.layers.Input(name='input_layer')(x)

        offset1 = tl.layers.Conv2d(18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset1')(net)
        cls.net1 = tl.layers.DeformableConv2d(32, (3, 3), act=tf.nn.relu, name='deformable1')(net, offset1)

        offset2 = tl.layers.Conv2d(18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset2')(cls.net1)
        cls.net2 = tl.layers.DeformableConv2d(64, (3, 3), act=tf.nn.relu, name='deformable2')(cls.net1, offset2)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_layer_n1(self):

        self.assertEqual(len(self.net1.all_layers), 3)
        self.assertEqual(len(self.net1.all_weights), 4)
        self.assertEqual(self.net1.count_weights(), 1400)
        self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [299, 299, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.net2.all_layers), 5)
        self.assertEqual(len(self.net2.all_weights), 8)
        self.assertEqual(self.net2.count_weights(), 25098)
        self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [299, 299, 64])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
