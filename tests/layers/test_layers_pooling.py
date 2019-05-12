#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from tests.utils import CustomTestCase


class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        ## 1D ========================================================================

        x_1_input_shape = [None, 100, 1]
        nin_1 = Input(x_1_input_shape, name='test_in1')

        n1 = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2, name='test_conv1d')(nin_1)
        n2 = tl.layers.MaxPool1d(filter_size=3, strides=2, padding='SAME', name='test_maxpool1d')(n1)
        n3 = tl.layers.MeanPool1d(filter_size=3, strides=2, padding='SAME', name='test_meanpool1d')(n1)
        n4 = tl.layers.GlobalMaxPool1d(name='test_maxpool1d')(n1)
        n5 = tl.layers.GlobalMeanPool1d(name='test_meanpool1d')(n1)
        n16 = tl.layers.MaxPool1d(filter_size=3, strides=1, padding='VALID', dilation_rate=2, name='test_maxpool1d')(n1)
        n17 = tl.layers.MeanPool1d(filter_size=3, strides=1, padding='VALID', dilation_rate=2,
                                   name='test_meanpool1d')(n1)

        cls.n1_shape = n1.get_shape().as_list()
        cls.n2_shape = n2.get_shape().as_list()
        cls.n3_shape = n3.get_shape().as_list()
        cls.n4_shape = n4.get_shape().as_list()
        cls.n5_shape = n5.get_shape().as_list()
        cls.n16_shape = n16.get_shape().as_list()
        cls.n17_shape = n17.get_shape().as_list()

        print("Printing Pool1d")
        print(nin_1._info[0].layer)
        print(n1._info[0].layer)
        print(n2._info[0].layer)
        print(n3._info[0].layer)
        print(n4._info[0].layer)
        print(n5._info[0].layer)
        print(n16._info[0].layer)
        print(n17._info[0].layer)

        ## 2D ========================================================================

        x_2_input_shape = [None, 100, 100, 3]
        nin_2 = Input(x_2_input_shape, name='test_in2')

        n6 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='test_conv2d')(nin_2)
        n7 = tl.layers.MaxPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME', name='test_maxpool2d')(n6)
        n8 = tl.layers.MeanPool2d(filter_size=(3, 3), strides=(2, 2), padding='SAME', name='test_meanpool2d')(n6)
        n9 = tl.layers.GlobalMaxPool2d(name='test_maxpool2d')(n6)
        n10 = tl.layers.GlobalMeanPool2d(name='test_meanpool2d')(n6)
        n15 = tl.layers.PoolLayer(name='test_pool2d')(n6)
        n18 = tl.layers.CornerPool2d('TopLeft', name='test_cornerpool2d')(n6)

        cls.n6_shape = n6.get_shape().as_list()
        cls.n7_shape = n7.get_shape().as_list()
        cls.n8_shape = n8.get_shape().as_list()
        cls.n9_shape = n9.get_shape().as_list()
        cls.n10_shape = n10.get_shape().as_list()
        cls.n15_shape = n15.get_shape().as_list()
        cls.n18_shape = n18.get_shape().as_list()

        print("Printing Pool2d")
        print(nin_2._info[0].layer)
        print(n6._info[0].layer)
        print(n7._info[0].layer)
        print(n8._info[0].layer)
        print(n9._info[0].layer)
        print(n10._info[0].layer)
        print(n15._info[0].layer)
        print(n18._info[0].layer)

        ## 3D ========================================================================

        x_3_input_shape = [None, 100, 100, 100, 3]
        nin_3 = Input(x_3_input_shape, name='test_in3')

        n11 = tl.layers.MeanPool3d(filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME',
                                   name='test_meanpool3d')(nin_3)
        n12 = tl.layers.GlobalMaxPool3d(name='test_maxpool3d')(nin_3)
        n13 = tl.layers.GlobalMeanPool3d(name='test_meanpool3d')(nin_3)
        n14 = tl.layers.MaxPool3d(filter_size=(3, 3, 3), strides=(2, 2, 2), padding='SAME',
                                  name='test_maxpool3d')(nin_3)

        cls.n11_shape = n11.get_shape().as_list()
        cls.n12_shape = n12.get_shape().as_list()
        cls.n13_shape = n13.get_shape().as_list()
        cls.n14_shape = n14.get_shape().as_list()

        print("Printing Pool3d")
        print(nin_3._info[0].layer)
        print(n11._info[0].layer)
        print(n12._info[0].layer)
        print(n13._info[0].layer)
        print(n14._info[0].layer)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

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
        self.assertEqual(self.n11_shape[1:5], [50, 50, 50, 3])

    def test_n12_shape(self):
        self.assertEqual(self.n12_shape[-1], 3)

    def test_n13_shape(self):
        self.assertEqual(self.n13_shape[-1], 3)

    def test_n14_shape(self):
        self.assertEqual(self.n14_shape[1:5], [50, 50, 50, 3])

    def test_n15_shape(self):
        self.assertEqual(self.n15_shape[1:4], [25, 25, 32])

    def test_n16_shape(self):
        self.assertEqual(self.n16_shape[1:4], [46, 32])

    def test_n17_shape(self):
        self.assertEqual(self.n17_shape[1:4], [48, 32])

    def test_n18_shape(self):
        self.assertEqual(self.n18_shape[1:], [50, 50, 32])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
