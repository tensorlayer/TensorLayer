#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *

from tests.utils import CustomTestCase


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 10, 10, 16]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.offset1 = tl.layers.Conv2d(n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME',
                                       name='offset1')(cls.input_layer)
        cls.deformconv1 = tl.layers.DeformableConv2d(
            offset_layer=cls.offset1, n_filter=32, filter_size=(3, 3), act=tf.nn.relu, name='deformable1'
        )(cls.input_layer)
        cls.offset2 = tl.layers.Conv2d(n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME',
                                       name='offset2')(cls.deformconv1)
        cls.deformconv2 = tl.layers.DeformableConv2d(
            offset_layer=cls.offset2, n_filter=64, filter_size=(3, 3), act=tf.nn.relu, name='deformable2'
        )(cls.deformconv1)

        cls.model = Model(cls.input_layer, cls.deformconv2)
        print("Testing Deformable Conv2d model: \n", cls.model)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):

        self.assertEqual(len(self.deformconv1._info[0].layer.all_weights), 2)
        self.assertEqual(self.deformconv1.get_shape().as_list()[1:], [10, 10, 32])

    def test_layer_n2(self):

        self.assertEqual(len(self.deformconv2._info[0].layer.all_weights), 2)
        self.assertEqual(self.deformconv2.get_shape().as_list()[1:], [10, 10, 64])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
