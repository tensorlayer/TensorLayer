#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl
from tests.utils import CustomTestCase

class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("\n#################################")

        self.batch_size = 5
        self.inputs_shape = [self.batch_size, 10, 10, 16]
        self.input_layer = tl.layers.Input(self.inputs_shape, name='input_layer')

        self.offset1 = tl.layers.Conv2d(n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME',
                                       name='offset1')(self.input_layer)
        self.init_deformconv1 = tl.layers.DeformableConv2d(
            offset_layer=self.offset1, n_filter=32, filter_size=(3, 3), act='relu', name='deformable1'
        )
        self.deformconv1 = self.init_deformconv1(self.input_layer)
        self.offset2 = tl.layers.Conv2d(n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME',
                                       name='offset2')(self.deformconv1)
        self.deformconv2 = tl.layers.DeformableConv2d(
            offset_layer=self.offset2, n_filter=64, filter_size=(3, 3), act='relu', name='deformable2'
        )(self.deformconv1)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n1(self):

        self.assertEqual(len(self.init_deformconv1.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.deformconv1)[1:], [10, 10, 32])

    def test_layer_n2(self):
        self.assertEqual(tl.get_tensor_shape(self.deformconv2)[1:], [10, 10, 64])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
