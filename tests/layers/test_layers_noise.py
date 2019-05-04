#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

from tests.utils import CustomTestCase


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 8
        cls.inputs_shape = [cls.batch_size, 200]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.dense = tl.layers.Dense(n_units=100, act=tf.nn.relu, in_channels=200)(cls.input_layer)

        cls.noiselayer = tl.layers.GaussianNoise(name='gaussian')(cls.dense)

        print("Testing GaussianNoise: \n", cls.noiselayer._info[0].layer)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        self.assertEqual(self.noiselayer.get_shape().as_list()[1:], [100])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
