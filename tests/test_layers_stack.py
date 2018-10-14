#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Stack_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float16, shape=[100, 30])

        cls.net_in = tl.layers.Input(name='input')(x)

        cls.net_drop = tl.layers.Dropout(keep=0.5, name='dropout')(cls.net_in, is_train=True)

        cls.net_d1 = tl.layers.Dense(n_units=10, name='dense1')(cls.net_drop)
        cls.net_d2 = tl.layers.Dense(n_units=10, name='dense2')(cls.net_drop)
        cls.net_d3 = tl.layers.Dense(n_units=10, name='dense3')(cls.net_drop)

        cls.net_stack = tl.layers.Stack(axis=1, name='stack')([cls.net_d1, cls.net_d2, cls.net_d3])

        cls.net_unstack = tl.layers.UnStack(axis=1, name='unstack')(cls.net_stack)

        cls.net_unstacked_d1 = cls.net_unstack.outputs[0]

        cls.net_unstacked_d2 = cls.net_unstack.outputs[1]

        cls.net_unstacked_d3 = cls.net_unstack.outputs[2]

        cls.all_layers = [
            cls.net_in,
            cls.net_drop,
            cls.net_d1,
            cls.net_d2,
            cls.net_d3,
            cls.net_stack,
            cls.net_unstack,
            cls.net_unstacked_d1,
            cls.net_unstacked_d2,
            cls.net_unstacked_d3,
        ]

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_objects_dtype(self):
        self.assertIsInstance(self.net_in, tl.layers.BuiltLayer)

        self.assertIsInstance(self.net_drop, tl.layers.BuiltLayer)

        self.assertIsInstance(self.net_d1, tl.layers.BuiltLayer)
        self.assertIsInstance(self.net_d2, tl.layers.BuiltLayer)
        self.assertIsInstance(self.net_d3, tl.layers.BuiltLayer)

        self.assertIsInstance(self.net_stack, tl.layers.BuiltLayer)

        self.assertIsInstance(self.net_unstack, tl.layers.BuiltLayer)

        self.assertIsInstance(self.net_unstacked_d1, tl.layers.BuiltLayer)
        self.assertIsInstance(self.net_unstacked_d2, tl.layers.BuiltLayer)
        self.assertIsInstance(self.net_unstacked_d3, tl.layers.BuiltLayer)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.net_in.local_drop), 0)

        self.assertEqual(len(self.net_drop.local_drop), 1)

        self.assertEqual(len(self.net_d1.local_drop), 0)
        self.assertEqual(len(self.net_d2.local_drop), 0)
        self.assertEqual(len(self.net_d3.local_drop), 0)

        self.assertEqual(len(self.net_stack.local_drop), 0)

        self.assertEqual(len(self.net_unstack.local_drop), 0)

        self.assertEqual(len(self.net_unstacked_d1.local_drop), 0)
        self.assertEqual(len(self.net_unstacked_d2.local_drop), 0)
        self.assertEqual(len(self.net_unstacked_d3.local_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.net_in.count_local_weights(), 0)

        self.assertEqual(self.net_drop.count_local_weights(), 0)

        self.assertEqual(self.net_d1.count_local_weights(), 310)
        self.assertEqual(self.net_d2.count_local_weights(), 310)
        self.assertEqual(self.net_d3.count_local_weights(), 310)

        self.assertEqual(self.net_stack.count_local_weights(), 0)

        self.assertEqual(self.net_unstack.count_local_weights(), 0)

        self.assertEqual(self.net_unstacked_d1.count_local_weights(), 0)
        self.assertEqual(self.net_unstacked_d2.count_local_weights(), 0)
        self.assertEqual(self.net_unstacked_d3.count_local_weights(), 0)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.net_in.local_weights), 0)

        self.assertEqual(len(self.net_drop.local_weights), 0)

        self.assertEqual(len(self.net_d1.local_weights), 2)
        self.assertEqual(len(self.net_d2.local_weights), 2)
        self.assertEqual(len(self.net_d3.local_weights), 2)

        self.assertEqual(len(self.net_stack.local_weights), 0)

        self.assertEqual(len(self.net_unstack.local_weights), 0)

        self.assertEqual(len(self.net_unstacked_d1.local_weights), 0)
        self.assertEqual(len(self.net_unstacked_d2.local_weights), 0)
        self.assertEqual(len(self.net_unstacked_d3.local_weights), 0)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer in self.all_layers:

                if layer.outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer.name, layer.outputs.dtype, tf.float16)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.net_in.outputs.shape, (100, 30))

        self.assertEqual(self.net_drop.outputs.shape, (100, 30))

        self.assertEqual(self.net_d1.outputs.shape, (100, 10))
        self.assertEqual(self.net_d2.outputs.shape, (100, 10))
        self.assertEqual(self.net_d3.outputs.shape, (100, 10))

        self.assertEqual(self.net_stack.outputs.shape, (100, 3, 10))

        self.assertEqual(self.net_unstack.outputs.shape, (3, 100, 10))

        self.assertEqual(self.net_unstacked_d1.outputs.shape, (100, 10))
        self.assertEqual(self.net_unstacked_d2.outputs.shape, (100, 10))
        self.assertEqual(self.net_unstacked_d3.outputs.shape, (100, 10))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
