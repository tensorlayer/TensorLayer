#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *

from tests.utils import CustomTestCase


class Layer_Core_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.batch_size = 8

        # ============== Layer ==============

        cls.base_layer = Layer(what=None)

        # ============== DenseLayer ==============

        inputs_shape = [None, 784]
        cls.innet = Input(inputs_shape)
        cls.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)(cls.innet)
        cls.dense2 = Dense(n_units=10, act=tf.nn.relu, b_init=None)(cls.dense1)
        cls.dense3 = Dense(n_units=10, act=tf.nn.relu, b_init=None)


    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):

        try:
            self.base_layer.build(None)
        except Exception as e:
            print(e)

        try:
            self.base_layer.forward(None)
        except Exception as e:
            print(e)

        try:
            self.base_layer[4] = 1
        except Exception as e:
            print(e)

        try:
            del self.base_layer[4]
        except Exception as e:
            print(e)

        try:
            Layer(what = 1)
        except Exception as e:
            print(e)


        print(self.innet)
        print(self.dense1)
        print(self.dense2)
        print(self.dense3)

        self.assertEqual(self.innet.weights, None)
        self.assertEqual(self.dense1.weights[0].get_shape().as_list(), [784, 800])
        self.assertEqual(self.dense1.weights[1].get_shape().as_list(), [800,])
        self.assertEqual(self.dense2.weights[0].get_shape().as_list(), [800, 10])
        self.assertEqual(len(self.dense1.weights), 2)
        self.assertEqual(len(self.dense2.weights), 1)

        self.model = Model(inputs=self.innet, outputs=self.dense2)
        self.results = self.model(np.ones(shape=(self.batch_size, 784)).astype(np.float32), is_train=True)

        self.assertEqual(len(self.model.weights), 3)

        self.assertEqual(self.innet._inputs_shape, [self.batch_size, 784])
        self.assertEqual(self.innet._outputs_shape, [self.batch_size, 784])
        self.assertEqual(self.dense1._inputs_shape, [self.batch_size, 784])
        self.assertEqual(self.dense1._outputs_shape, [self.batch_size, 800])
        self.assertEqual(self.dense2._inputs_shape, [self.batch_size, 800])
        self.assertEqual(self.dense2._outputs_shape, [self.batch_size, 10])
        self.assertIsInstance(self.results, Layer)
        self.assertIsInstance(self.results.outputs, tf.Tensor)
        self.assertEqual(self.results._outputs_shape, [self.batch_size, 10])

        print(self.innet)
        print(self.dense1)
        print(self.dense2)
        print(self.dense3)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
