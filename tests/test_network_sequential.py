#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

try:
    from tests.unittests_helper import CustomTestCase
except ImportError:
    from unittests_helper import CustomTestCase


class Network_Sequential_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            cls.model = tl.networks.Sequential(name="My_Seq_net")

            cls.model.add(tl.layers.ExpandDimsLayer(axis=1, name="expand_layer_1"))
            cls.model.add(tl.layers.FlattenLayer(name="flatten_layer_1"))

            cls.model.add(tl.layers.ExpandDimsLayer(axis=2, name="expand_layer_2"))
            cls.model.add(tl.layers.TileLayer(multiples=[1, 1, 3], name="tile_layer_2"))
            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 16, 6], name="reshape_layer_2"))
            cls.model.add(tl.layers.TransposeLayer(perm=[0, 2, 1], name='transpose_layer_2'))
            cls.model.add(tl.layers.FlattenLayer(name="flatten_layer_2"))

            cls.model.add(tl.layers.DenseLayer(n_units=10, act=tf.nn.relu, name="seq_layer_1"))

            cls.model.add(tl.layers.DenseLayer(n_units=20, act=None, name="seq_layer_2"))
            cls.model.add(tl.layers.PReluLayer(channel_shared=True, name="prelu_layer_2"))

            cls.model.add(tl.layers.DenseLayer(n_units=30, act=None, name="seq_layer_3"))
            cls.model.add(tl.layers.PReluLayer(channel_shared=False, name="prelu_layer_3"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=None, name="seq_layer_4"))
            cls.model.add(tl.layers.PRelu6Layer(channel_shared=True, name="prelu6_layer_4"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=None, name="seq_layer_5"))
            cls.model.add(tl.layers.PRelu6Layer(channel_shared=False, name="prelu6_layer_5"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=None, name="seq_layer_6"))
            cls.model.add(tl.layers.PTRelu6Layer(channel_shared=True, name="ptrelu6_layer_6"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=None, name="seq_layer_7"))
            cls.model.add(tl.layers.PTRelu6Layer(channel_shared=False, name="ptrelu6_layer_7"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=tf.nn.relu, name="seq_layer_8"))
            cls.model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=True, name="dropout_layer_8"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=tf.nn.relu, name="seq_layer_9"))
            cls.model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=False, name="dropout_layer_9"))

            cls.model.add(tl.layers.SlimNetsLayer(slim_layer=slim.fully_connected, slim_args={'num_outputs': 50, 'activation_fn': None}, act=tf.nn.relu, name="seq_layer_10"))

            cls.model.add(tl.layers.KerasLayer(keras_layer=keras.layers.Dense, keras_args={'units': 50}, act=tf.nn.relu, name="seq_layer_11"))

            plh = tf.placeholder(tf.float16, (100, 32))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_get_all_param_tensors(self):
        n_weight_tensors = len(self.model.get_all_params())
        tl.logging.debug("# Weight Tensors: %d" % n_weight_tensors)

        self.assertEqual(n_weight_tensors, 30)

    def test_get_all_drop_plh(self):
        n_drop_plh = len(self.model.all_drop)
        tl.logging.debug("# Dropout Placeholders: %d" % n_drop_plh)

        self.assertEqual(n_drop_plh, 1)

    def test_count_params(self):
        n_params = self.model.count_params()
        tl.logging.debug("# Parameters: %d" % n_params)

        self.assertEqual(n_params, 18574)

    def test__getitem__(self):

        with self.assertNotRaises(Exception):

            layer = self.model["seq_layer_1"]
            self.assertTrue(isinstance(layer, tl.layers.DenseLayer))

    def test_network_output(self):
        self.assertEqual(self.train_model.shape, (100, 50))
        self.assertEqual(self.test_model.shape, (100, 50))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
