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
        cls.dropout1 = Dropout(keep=0.8)(cls.dense1)
        cls.dense2 = Dense(n_units=10, act=tf.nn.relu, b_init=None)(cls.dropout1)
        cls.dense3 = Dense(n_units=10, act=tf.nn.relu, b_init=None)

        print(cls.innet)
        print(cls.dense1)
        print(cls.dropout1)
        print(cls.dense2)
        print(cls.dense3)

        cls.model = Model(inputs=cls.innet, outputs=cls.dense2)
        cls.results_train = cls.model(np.ones(shape=(cls.batch_size, 784)).astype(np.float32), is_train=True)
        cls.results_test = cls.model(np.ones(shape=(cls.batch_size, 784)).astype(np.float32), is_train=False)


    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):

        # test exceptional cases
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

    def test_net2(self):

        # test weights
        self.assertEqual(self.innet.weights, None)
        self.assertEqual(self.dropout1.weights, None)
        self.assertEqual(self.dense1.weights[0].get_shape().as_list(), [784, 800])
        self.assertEqual(self.dense1.weights[1].get_shape().as_list(), [800,])
        self.assertEqual(self.dense2.weights[0].get_shape().as_list(), [800, 10])
        self.assertEqual(len(self.dense1.weights), 2)
        self.assertEqual(len(self.dense2.weights), 1)

        self.assertEqual(len(self.model.weights), 3)

        # test input output
        self.assertEqual(self.innet._inputs_shape, [self.batch_size, 784])
        self.assertEqual(self.innet._outputs_shape, [self.batch_size, 784])
        self.assertEqual(self.dense1._inputs_shape, [self.batch_size, 784])
        self.assertEqual(self.dense1._outputs_shape, [self.batch_size, 800])
        self.assertEqual(self.dense2._inputs_shape, [self.batch_size, 800])
        self.assertEqual(self.dense2._outputs_shape, [self.batch_size, 10])

        self.assertEqual(self.results_train.get_shape().as_list(), [self.batch_size, 10])
        self.assertEqual(self.results_test.get_shape().as_list(), [self.batch_size, 10])

        # test printing
        print(self.innet)
        print(self.dense1)
        print(self.dropout1)
        print(self.dense2)
        print(self.dense3)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
