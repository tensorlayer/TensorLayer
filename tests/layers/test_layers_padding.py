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
        cls.input_layer1 = tl.layers.Input([None, 100, 1], name='input_layer1')

        n1 = tl.layers.ZeroPad1d(padding=1)(cls.input_layer1)
        n2 = tl.layers.ZeroPad1d(padding=(2, 3))(cls.input_layer1)

        print(n1._info[0].layer)
        print(n2._info[0].layer)

        cls.n1_shape = n1.get_shape().as_list()
        cls.n2_shape = n2.get_shape().as_list()

        ## 2D
        cls.input_layer2 = tl.layers.Input([None, 100, 100, 3], name='input_layer2')

        n0 = tl.layers.PadLayer([[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')(cls.input_layer2)
        n3 = tl.layers.ZeroPad2d(padding=2)(cls.input_layer2)
        n4 = tl.layers.ZeroPad2d(padding=(2, 3))(cls.input_layer2)
        n5 = tl.layers.ZeroPad2d(padding=((3, 3), (4, 4)))(cls.input_layer2)

        print(n0._info[0].layer)
        print(n3._info[0].layer)
        print(n4._info[0].layer)
        print(n5._info[0].layer)

        cls.n0_shape = n0.get_shape().as_list()
        print(cls.n0_shape)
        cls.n3_shape = n3.get_shape().as_list()
        cls.n4_shape = n4.get_shape().as_list()
        cls.n5_shape = n5.get_shape().as_list()

        ## 3D
        cls.input_layer3 = tl.layers.Input([None, 100, 100, 100, 3], name='input_layer3')

        n6 = tl.layers.ZeroPad3d(padding=2)(cls.input_layer3)
        n7 = tl.layers.ZeroPad3d(padding=(2, 3, 4))(cls.input_layer3)
        n8 = tl.layers.ZeroPad3d(padding=((3, 3), (4, 4), (5, 5)))(cls.input_layer3)

        print(n6._info[0].layer)
        print(n7._info[0].layer)
        print(n8._info[0].layer)

        cls.n6_shape = n6.get_shape().as_list()
        cls.n7_shape = n7.get_shape().as_list()
        cls.n8_shape = n8.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_n0_shape(self):
        self.assertEqual(self.n0_shape[1:], [106, 106, 3])

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

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
