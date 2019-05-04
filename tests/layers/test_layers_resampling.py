#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/wurundi/workspace/tensorlayer2")

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

        ## 2D ========================================================================

        x_2_input_shape = [None, 100, 100, 3]
        nin_2 = Input(x_2_input_shape)

        n6 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='test_conv2d')(nin_2)

        n7 = tl.layers.UpSampling2d(scale=(2, 2), name='test_UpSampling2d_1')(n6)

        n8 = tl.layers.UpSampling2d(scale=3, name='test_UpSampling2d_2')(n6)

        n9 = tl.layers.DownSampling2d(scale=(2, 2), name='test_DownSampling2d_1')(n6)

        n10 = tl.layers.DownSampling2d(scale=5, name='test_DownSampling2d_2')(n6)

        cls.n6_shape = n6.get_shape().as_list()
        cls.n7_shape = n7.get_shape().as_list()
        cls.n8_shape = n8.get_shape().as_list()
        cls.n9_shape = n9.get_shape().as_list()
        cls.n10_shape = n10.get_shape().as_list()

        print("Printing UpSampling2d")
        print(nin_2._info[0].layer)
        print(n6._info[0].layer)
        print(n7._info[0].layer)
        print(n8._info[0].layer)
        print(n9._info[0].layer)
        print(n10._info[0].layer)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_UpSampling2d(self):
        self.assertEqual(self.n7_shape[1:3], [100, 100])
        self.assertEqual(self.n8_shape[1:3], [150, 150])

        try:
            layer = tl.layers.UpSampling2d(scale=(2, 2, 2))
        except Exception as e:
            print(e)

    def test_DownSampling2d(self):
        self.assertEqual(self.n9_shape[1:3], [25, 25])
        self.assertEqual(self.n10_shape[1:3], [10, 10])

        try:
            layer = tl.layers.DownSampling2d(scale=(2, 2, 2))
        except Exception as e:
            print(e)


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
