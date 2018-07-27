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

            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, 16, 16, 16, 1], name="reshape_layer_1"))

            cls.model.add(tl.layers.PadLayer(padding=[[0, 0], [4, 4], [3, 3], [2, 2], [0, 0]], name='pad_layer_2'))
            cls.model.add(tl.layers.ZeroPad3d(padding=2, name='zeropad3d_layer_2-1'))
            cls.model.add(tl.layers.ZeroPad3d(padding=(2, 2, 2), name='zeropad3d_layer_2-2'))
            cls.model.add(tl.layers.ZeroPad3d(padding=((2, 2), (3, 3), (4, 4)), name='zeropad3d_layer_2-3'))
            cls.model.add(tl.layers.ScaleLayer(init_scale=2., name='scale_layer_12'))

            plh = tf.placeholder(tf.float16, (100, 16, 16, 16))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.model.all_drop), 0)

    def test_count_params(self):
        self.assertEqual(self.model.count_params(), 1)

    def test_count_param_tensors(self):
        self.assertEqual(len(self.model.get_all_params()), 1)

    def test_count_layers(self):
        self.assertEqual(self.model.count_layers(), 7)

    def test_network_shapes(self):

        self.assertEqual(self.model["input_layer"].outputs.shape, (100, 16, 16, 16))

        self.assertEqual(self.model["reshape_layer_1"].outputs.shape, (100, 16, 16, 16, 1))

        self.assertEqual(self.model["pad_layer_2"].outputs.shape, (100, 24, 22, 20, 1))
        self.assertEqual(self.model["zeropad3d_layer_2-1"].outputs.shape, (100, 28, 26, 24, 1))
        self.assertEqual(self.model["zeropad3d_layer_2-2"].outputs.shape, (100, 32, 30, 28, 1))
        self.assertEqual(self.model["zeropad3d_layer_2-3"].outputs.shape, (100, 36, 36, 36, 1))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
