#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Merge_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_concat(self):

        class CustomModel(tl.models.Model):

            def __init__(self):
                super(CustomModel, self).__init__()
                self.dense1 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu1_1')
                self.dense2 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu2_1')
                self.concat = tl.layers.Concat(concat_dim=1, name='concat_layer')

            def forward(self, inputs):
                d1 = self.dense1(inputs)
                d2 = self.dense2(inputs)
                outputs = self.concat([d1, d2])
                return outputs

        model = CustomModel()
        model.train()
        inputs = tf.convert_to_tensor(np.random.random([4, 20]).astype(np.float32))
        outputs = model(inputs)
        print(model)

        self.assertEqual(outputs.get_shape().as_list(), [4, 20])

    def test_elementwise(self):

        class CustomModel(tl.models.Model):

            def __init__(self):
                super(CustomModel, self).__init__()
                self.dense1 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu1_1')
                self.dense2 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu2_1')
                self.element = tl.layers.Elementwise(combine_fn=tf.minimum, name='minimum', act=tf.identity)

            def forward(self, inputs):
                d1 = self.dense1(inputs)
                d2 = self.dense2(inputs)
                outputs = self.element([d1, d2])
                return outputs, d1, d2

        model = CustomModel()
        model.train()
        inputs = tf.convert_to_tensor(np.random.random([4, 20]).astype(np.float32))
        outputs, d1, d2 = model(inputs)
        print(model)

        min = tf.minimum(d1, d2)
        self.assertEqual(outputs.get_shape().as_list(), [4, 10])
        self.assertTrue(np.array_equal(min.numpy(), outputs.numpy()))


if __name__ == '__main__':

    unittest.main()
