#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Core_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):

        self.batch_size = 8

        self.inputs_shape = [self.batch_size, 784]
        self.input = tl.layers.Input(self.inputs_shape)
        self.dense1 = tl.layers.Dense(n_units=800, act=tl.ReLU, in_channels=784, name='test_dense')
        self.n1 = self.dense1(self.input)

        self.dropout1 = tl.layers.Dropout(keep=0.8)
        self.n2 = self.dropout1(self.n1)

        self.dense2 = tl.layers.Dense(n_units=10, act='relu', b_init=None, in_channels=800)
        self.n3 = self.dense2(self.n2)

        self.dense3 = tl.layers.Dense(n_units=10, act='relu', b_init=None, in_channels=10)
        self.n4 = self.dense3(self.n3)

        self.concat = tl.layers.Concat(concat_dim=-1)([self.n2, self.n3])

        class get_model(tl.layers.Module):
            def __init__(self):
                super(get_model, self).__init__()
                self.layer1 = tl.layers.Dense(n_units=800, act=tl.ReLU, in_channels=784, name='test_dense')
                self.dp = tl.layers.Dropout(keep=0.8)
                self.layer2 = tl.layers.Dense(n_units=10, act='relu', b_init=None, in_channels=800)
                self.layer3 = tl.layers.Dense(n_units=10, act='relu', b_init=None, in_channels=10)

            def forward(self, inputs):
                z = self.layer1(inputs)
                z = self.dp(z)
                z = self.layer2(z)
                z = self.layer3(z)
                return z

        self.net = get_model()


    @classmethod
    def tearDownClass(cls):
        pass

    def test_dense(self):
        self.assertEqual(tl.get_tensor_shape(self.n1), [self.batch_size, 800])

    def test_dense_nonbias(self):
        self.assertEqual(len(self.dense2.all_weights), 1)

    def test_dropout(self):
        self.assertEqual(len(self.dropout1.all_weights), 0)

    def test_model(self):
        self.assertEqual(len(self.net.all_weights), 4)


if __name__ == '__main__':

    unittest.main()
