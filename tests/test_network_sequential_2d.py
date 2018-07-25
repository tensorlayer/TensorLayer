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

            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 16, 16, 1], name="reshape_layer_1"))

            cls.model.add(tl.layers.UpSampling2dLayer(size=(2, 2), is_scale=True, method=0, align_corners=True, name="upsample2d_layer_2"))
            cls.model.add(tl.layers.DownSampling2dLayer(size=(2, 2), is_scale=True, method=0, align_corners=True, name="downsample2d_layer_2"))

            plh = tf.placeholder(tf.float16, (100, 16, 16))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_get_all_param_tensors(self):
        self.assertEqual(len(self.model.get_all_params()), 0)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.model.all_drop), 0)

    def test_count_params(self):
        self.assertEqual(self.model.count_params(), 0)

    def test_count_layers(self):
        self.assertEqual(self.model.count_layers(), 4)

    def test_network_shapes(self):

        self.assertEqual(self.model["input_layer"].outputs.shape, (100, 16, 16))

        self.assertEqual(self.model["reshape_layer_1"].outputs.shape, (100, 16, 16, 1))

        self.assertEqual(self.model["upsample2d_layer_2"].outputs.shape, (100, 32, 32, 1))
        self.assertEqual(self.model["downsample2d_layer_2"].outputs.shape, (100, 16, 16, 1))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
