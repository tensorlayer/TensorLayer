#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        ## 1D ========================================================================

        x_1 = tf.placeholder(tf.float32, (None, 100, 1))
        nin_1 = tl.layers.InputLayer(x_1, name='in1')

        n1 = tl.layers.Conv1d(nin_1, n_filter=32, filter_size=5, stride=2, name='conv1d')
        n2 = tl.layers.MaxPool1d(n1, filter_size=3, strides=2, padding='same', name='maxpool1d')
        n3 = tl.layers.MeanPool1d(n1, filter_size=3, strides=2, padding='same', name='meanpool1d')
        n4 = tl.layers.GlobalMaxPool1d(n1, name='maxpool1d')
        n5 = tl.layers.GlobalMeanPool1d(n1, name='meanpool1d')

        cls.n1_shape = n1.outputs.get_shape().as_list()
        cls.n2_shape = n2.outputs.get_shape().as_list()
        cls.n3_shape = n3.outputs.get_shape().as_list()
        cls.n4_shape = n4.outputs.get_shape().as_list()
        cls.n5_shape = n5.outputs.get_shape().as_list()

        ## 2D ========================================================================

        x_2 = tf.placeholder(tf.float32, (None, 100, 100, 3))
        nin_2 = tl.layers.InputLayer(x_2, name='in2')

        n6 = tl.layers.Conv2d(nin_2, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='conv2d')
        n7 = tl.layers.MaxPool2d(n6, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='maxpool2d')
        n8 = tl.layers.MeanPool2d(n6, filter_size=(3, 3), strides=(2, 2), padding='SAME', name='meanpool2d')
        n9 = tl.layers.GlobalMaxPool2d(n6, name='maxpool2d')
        n10 = tl.layers.GlobalMeanPool2d(n6, name='meanpool2d')

        cls.n6_shape = n6.outputs.get_shape().as_list()
        cls.n7_shape = n7.outputs.get_shape().as_list()
        cls.n8_shape = n8.outputs.get_shape().as_list()
        cls.n9_shape = n9.outputs.get_shape().as_list()
        cls.n10_shape = n10.outputs.get_shape().as_list()

        ## 3D ========================================================================

        x_3 = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
        nin_3 = tl.layers.InputLayer(x_3, name='in')

        n11 = tl.layers.MeanPool3d(nin_3, filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME', name='meanpool3d')
        n12 = tl.layers.GlobalMaxPool3d(nin_3)
        n13 = tl.layers.GlobalMeanPool3d(nin_3)

        cls.n11_shape = n11.outputs.get_shape().as_list()
        cls.n12_shape = n12.outputs.get_shape().as_list()
        cls.n13_shape = n13.outputs.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_n1_shape(self):
        self.assertEqual(self.n1_shape[1:3], [50, 32])

    def test_n2_shape(self):
        self.assertEqual(self.n2_shape[1:3], [25, 32])

    def test_n3_shape(self):
        self.assertEqual(self.n3_shape[1:3], [25, 32])

    def test_n4_shape(self):
        self.assertEqual(self.n4_shape[-1], 32)

    def test_n5_shape(self):
        self.assertEqual(self.n5_shape[-1], 32)

    def test_n6_shape(self):
        self.assertEqual(self.n6_shape[1:4], [50, 50, 32])

    def test_n7_shape(self):
        self.assertEqual(self.n7_shape[1:4], [25, 25, 32])

    def test_n8_shape(self):
        self.assertEqual(self.n8_shape[1:4], [25, 25, 32])

    def test_n9_shape(self):
        self.assertEqual(self.n9_shape[-1], 32)

    def test_n10_shape(self):
        self.assertEqual(self.n10_shape[-1], 32)

    def test_n11_shape(self):
        self.assertEqual(self.n11_shape, [None, 50, 50, 50, 3])

    def test_n12_shape(self):
        self.assertEqual(self.n12_shape, [None, 3])

    def test_n13_shape(self):
        self.assertEqual(self.n13_shape, [None, 3])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
