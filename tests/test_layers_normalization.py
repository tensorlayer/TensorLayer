#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


def model(x, is_train=True, reuse=False):

    with tf.variable_scope("model", reuse=reuse):
        n = tl.layers.Input(name='in')(x)
        n = tl.layers.Conv2d(n_filter=80, name='conv2d_1')(n)
        n = tl.layers.BatchNorm(n, name='norm_batch')(n, is_train=is_train)
        n = tl.layers.Conv2d(n_filter=80, name='conv2d_2')(n)
        n = tl.layers.LocalResponseNormLayer(name='norm_local')(n)
        n = tl.layers.LayerNormLayer(reuse=reuse, name='norm_layer')(n)
        n = tl.layers.InstanceNormLayer(name='norm_instance')(n)

        n.outputs = tf.reshape(n.outputs, [-1, 80, 100, 100])
        n = tl.layers.GroupNormLayer(groups=40, data_format='channels_first', name='groupnorm')(n)

        n.outputs = tf.reshape(n.outputs, [-1, 100, 100, 80])
        n = tl.layers.SwitchNormLayer(name='switchnorm')(n)

        n = tl.layers.QuantizedConv2dWithBN(n_filter=3, name='quan_cnn_with_bn')(n, is_train=is_train)

        n = tl.layers.FlattenLayer(name='flatten')(n)
        n = tl.layers.QuantizedDenseWithBN(n_units=10, name='quan_dense_with_bn')(n)

    return n


class Layer_Normalization_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, [None, 100, 100, 3])

        net_train = model(x, is_train=True, reuse=False)
        net_eval = model(x, is_train=False, reuse=True)

        net_train.print_layers()
        net_train.print_params(False)

        cls.data = dict()
        cls.data["train_network"] = dict()
        cls.data["eval_network"] = dict()

        cls.data["train_network"]["layers"] = net_train.all_layers
        cls.data["eval_network"]["layers"] = net_eval.all_layers

        cls.data["train_network"]["params"] = net_train.all_weights

        cls.data["train_network"]["n_params"] = net_train.count_weights()

        print(net_train.count_weights())

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_all_layers(self):
        self.assertEqual(len(self.data["train_network"]["layers"]), 12)
        self.assertEqual(len(self.data["eval_network"]["layers"]), 12)

    def test_all_params(self):
        self.assertEqual(len(self.data["train_network"]["params"]), 28)

    def test_n_params(self):
        self.assertEqual(self.data["train_network"]["n_params"], 363098)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
