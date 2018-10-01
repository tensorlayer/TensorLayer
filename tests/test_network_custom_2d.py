#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):

            def fire_module(inputs, squeeze_depth, expand_depth, name):
                """Fire module: squeeze input filters, then apply spatial convolutions."""

                with tf.variable_scope(name):
                    squeezed = tl.layers.Conv2d(
                        n_filter=squeeze_depth,
                        filter_size=(1, 1),
                        strides=(1, 1),
                        padding='SAME',
                        act=tf.nn.relu,
                        name='squeeze'
                    )(inputs)

                    e1x1 = tl.layers.Conv2d(
                        n_filter=expand_depth,
                        filter_size=(1, 1),
                        strides=(1, 1),
                        padding='SAME',
                        act=tf.nn.relu,
                        name='e1x1'
                    )(squeezed)

                    e3x3 = tl.layers.Conv2d(
                        n_filter=expand_depth,
                        filter_size=(3, 3),
                        strides=(1, 1),
                        padding='SAME',
                        act=tf.nn.relu,
                        name='e3x3'
                    )(squeezed)

                    return tl.layers.ConcatLayer(concat_dim=3, name='concat')([e1x1, e3x3])

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):
                    input_layer = tl.layers.InputLayer(name='input_layer')

                    net = fire_module(input_layer, 32, 24, "fire_module_1")
                    net = fire_module(net, 32, 24, "fire_module_2")

                    return input_layer, net

            cls.model = MyCustomNetwork(name="my_custom_network")

            plh = tf.placeholder(tf.float16, (100, 16, 16, 3))

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
        self.assertEqual(self.train_model.count_weights(), 17152)
        self.assertEqual(self.test_model.count_weights(), 17152)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 17152)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 12)
        self.assertEqual(len(self.test_model.get_all_weights()), 12)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 12)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 9)
        self.assertEqual(self.test_model.count_layers(), 9)
        self.assertEqual(self.model.count_layers(), 9)

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

        self.assertEqual(self.train_model["input_layer"].outputs.shape, (100, 16, 16, 3))
        self.assertEqual(self.test_model["input_layer"].outputs.shape, (100, 16, 16, 3))

        self.assertEqual(self.train_model["fire_module_1/squeeze"].outputs.shape, (100, 16, 16, 32))
        self.assertEqual(self.test_model["fire_module_1/squeeze"].outputs.shape, (100, 16, 16, 32))

        self.assertEqual(self.train_model["fire_module_1/e1x1"].outputs.shape, (100, 16, 16, 24))
        self.assertEqual(self.test_model["fire_module_1/e1x1"].outputs.shape, (100, 16, 16, 24))

        self.assertEqual(self.train_model["fire_module_1/e3x3"].outputs.shape, (100, 16, 16, 24))
        self.assertEqual(self.test_model["fire_module_1/e3x3"].outputs.shape, (100, 16, 16, 24))

        self.assertEqual(self.train_model["fire_module_1/concat"].outputs.shape, (100, 16, 16, 48))
        self.assertEqual(self.test_model["fire_module_1/concat"].outputs.shape, (100, 16, 16, 48))

        self.assertEqual(self.train_model["fire_module_2/squeeze"].outputs.shape, (100, 16, 16, 32))
        self.assertEqual(self.test_model["fire_module_2/squeeze"].outputs.shape, (100, 16, 16, 32))

        self.assertEqual(self.train_model["fire_module_2/e1x1"].outputs.shape, (100, 16, 16, 24))
        self.assertEqual(self.test_model["fire_module_2/e1x1"].outputs.shape, (100, 16, 16, 24))

        self.assertEqual(self.train_model["fire_module_2/e3x3"].outputs.shape, (100, 16, 16, 24))
        self.assertEqual(self.test_model["fire_module_2/e3x3"].outputs.shape, (100, 16, 16, 24))

        self.assertEqual(self.train_model["fire_module_2/concat"].outputs.shape, (100, 16, 16, 48))
        self.assertEqual(self.test_model["fire_module_2/concat"].outputs.shape, (100, 16, 16, 48))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
