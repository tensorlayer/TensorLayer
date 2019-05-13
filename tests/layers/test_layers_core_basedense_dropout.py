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

        cls.inputs_shape = [None, 784]
        cls.innet = Input(cls.inputs_shape)
        cls.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784, name='test_dense')(cls.innet)
        cls.dropout1 = Dropout(keep=0.8)(cls.dense1)
        cls.dense2 = Dense(n_units=10, act=tf.nn.relu, b_init=None)(cls.dropout1)
        cls.dense3 = Dense(n_units=10, act=tf.nn.relu, b_init=None)
        cls.concat = Concat(concat_dim=-1)([cls.dense2, cls.dropout1])

        cls.model = Model(inputs=cls.innet, outputs=cls.dense2)

    @classmethod
    def tearDownClass(cls):
        pass

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
            Layer(what=1)
        except Exception as e:
            print(e)

    def test_net2(self):

        # test weights
        self.assertEqual(self.innet._info[0].layer.all_weights, [])
        self.assertEqual(self.dropout1._info[0].layer.all_weights, [])
        self.assertEqual(self.dense1._info[0].layer.all_weights[0].get_shape().as_list(), [784, 800])
        self.assertEqual(self.dense1._info[0].layer.all_weights[1].get_shape().as_list(), [
            800,
        ])
        self.assertEqual(self.dense2._info[0].layer.all_weights[0].get_shape().as_list(), [800, 10])
        self.assertEqual(len(self.dense1._info[0].layer.all_weights), 2)
        self.assertEqual(len(self.dense2._info[0].layer.all_weights), 1)

        self.assertEqual(len(self.model.all_weights), 3)

        # a special case
        self.model.release_memory()

        # test printing
        # print(self.innet)
        # print(self.dense1)
        # print(self.dropout1)
        # print(self.dense2)
        # print(self.dense3)

    def test_special_cases(self):
        try:
            innet = Input([121])
            dense1 = Dense(n_units=800, act=tf.nn.relu)(innet)
        except Exception as e:
            print(e)

    def test_modellayer(self):

        data = np.random.normal(size=[self.batch_size, self.inputs_shape[1]]).astype(np.float32)

        origin_results_train = self.model(data, is_train=True)
        origin_results_test = self.model(data, is_train=False)

        new_innet = Input(self.inputs_shape)
        new_mlayer = ModelLayer(self.model)(new_innet)

        newmodel = Model(inputs=new_innet, outputs=new_mlayer)

        new_results_train = newmodel(data, is_train=True)
        new_results_test = newmodel(data, is_train=False)

        self.assertEqual(origin_results_train.shape, new_results_train.shape)
        self.assertTrue(np.array_equal(origin_results_test.shape, new_results_test.shape))

        newmodel.release_memory()

    def test_layerlist(self):
        innet = Input(self.inputs_shape)
        hlayer = LayerList(
            [
                ModelLayer(self.model),
                LayerList([Dense(n_units=100), Dense(n_units=10)]),
                Dense(n_units=5),
                Dense(n_units=4)
            ]
        )(innet)
        model = Model(inputs=innet, outputs=hlayer)

        # for w in model.all_weights:
        #     print(w.name)

        data = np.random.normal(size=[self.batch_size, self.inputs_shape[1]]).astype(np.float32)
        pred = model(data, is_train=False)
        self.assertEqual(pred.get_shape().as_list(), [self.batch_size, 4])

        print(model)

        model.release_memory()

    def test_duplicate_names(self):
        dense1 = tl.layers.Dense(n_units=10, name='test_densehh')
        print(dense1)
        try:
            dense2 = tl.layers.Dense(n_units=10, name='test_densehh')
            print(dense2)
        except Exception as e:
            print(e)
        dense1 = tl.layers.Dense(n_units=10, name='test_densehh1')
        dense2 = tl.layers.Dense(n_units=10, name='test_densehh2')
        print(dense1)
        print(dense2)

    def test_dropout(self):
        data_x = np.random.random([10, 784]).astype(np.float32)
        pred_y_1 = self.model(data_x, is_train=True)
        pred_y_2 = self.model(data_x, is_train=True)
        self.assertFalse(np.allclose(pred_y_1, pred_y_2))
        pred_y_1 = self.model(data_x, is_train=False)
        pred_y_2 = self.model(data_x, is_train=False)
        self.assertTrue(np.allclose(pred_y_1, pred_y_2))


if __name__ == '__main__':

    unittest.main()
