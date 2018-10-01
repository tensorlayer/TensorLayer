#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):
                    input_layer = tl.layers.InputLayer(name='input_layer')

                    net_0 = tl.layers.DenseLayer(n_units=24, act=tf.nn.relu, name='dense_layer_1')(input_layer)
                    net_1 = tl.layers.DenseLayer(n_units=24, act=tf.nn.relu, name='dense_layer_2')(input_layer)

                    net = tl.layers.ElementwiseLayer(combine_fn=tf.minimum, name='elementwise_layer_3')([net_0, net_1])

                    return input_layer, net

            cls.model = MyCustomNetwork(name="my_custom_network")

            plh = tf.placeholder(tf.float16, (100, 30))

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

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
        self.assertEqual(self.train_model.count_weights(), 1488)
        self.assertEqual(self.test_model.count_weights(), 1488)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 1488)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 4)
        self.assertEqual(len(self.test_model.get_all_weights()), 4)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 4)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 4)
        self.assertEqual(self.test_model.count_layers(), 4)
        self.assertEqual(self.model.count_layers(), 4)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float16)
                    )

                if self.test_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float16)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_layer"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["input_layer"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["dense_layer_1"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["dense_layer_1"].outputs.shape, (100, 24))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 24))

        self.assertEqual(self.train_model["elementwise_layer_3"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["elementwise_layer_3"].outputs.shape, (100, 24))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
