apt#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_Multiple_Outputs_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):

                    data_plh_1 = tl.layers.InputLayer(name="data_plh_1")
                    data_plh_2 = tl.layers.InputLayer(name="data_plh_2")

                    network_1 = tl.layers.DenseLayer(n_units=20, name="dense_layer_1")(data_plh_1)
                    network_2 = tl.layers.DenseLayer(n_units=10, name="dense_layer_2")(data_plh_2)

                    network = tl.layers.ConcatLayer(concat_dim=1, name='concat_layer_3')([network_1, network_2])

                    return (data_plh_1, data_plh_2), network

            cls.model = MyCustomNetwork(name="my_custom_network")

            plh_1 = tf.placeholder(tf.float16, shape=(100, 50))
            plh_2 = tf.placeholder(tf.float16, shape=(100, 30))

            cls.train_model = cls.model.build((plh_1, plh_2), reuse=False, is_train=True)
            cls.test_model = cls.model.build((plh_1, plh_2), reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 1330)
        self.assertEqual(self.test_model.count_weights(), 1330)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 1330)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 4)
        self.assertEqual(len(self.test_model.get_all_weights()), 4)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 4)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 5)
        self.assertEqual(self.test_model.count_layers(), 5)
        self.assertEqual(self.model.count_layers(), 5)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float16)
                    )

            for layer_name in self.test_model.all_layers:

                if self.test_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float16)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["data_plh_1"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["data_plh_1"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["data_plh_2"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["data_plh_2"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["dense_layer_1"].outputs.shape, (100, 20))
        self.assertEqual(self.test_model["dense_layer_1"].outputs.shape, (100, 20))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 10))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
