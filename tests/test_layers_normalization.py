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
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.Conv2d(n, n_filter=80, name='conv2d_1')
        n = tl.layers.BatchNormLayer(n, is_train=is_train, name='norm_batch')
        n = tl.layers.Conv2d(n, n_filter=80, name='conv2d_2')
        n = tl.layers.LocalResponseNormLayer(n, name='norm_local')
        n = tl.layers.LayerNormLayer(n, reuse=reuse, name='norm_layer')
        n = tl.layers.InstanceNormLayer(n, name='norm_instance')
        # n = tl.layers.GroupNormLayer(n, groups=40, name='groupnorm')
        n.outputs = tf.reshape(n.outputs, [-1, 80, 100, 100])
        n = tl.layers.GroupNormLayer(n, groups=40, data_format='channels_first', name='groupnorm')
        n.outputs = tf.reshape(n.outputs, [-1, 100, 100, 80])
        n = tl.layers.SwitchNormLayer(n, name='switchnorm')
        n = tl.layers.QuanConv2dWithBN(n, n_filter=3, is_train=is_train, name='quan_cnn_with_bn')
        n = tl.layers.FlattenLayer(n, name='flatten')
        n = tl.layers.QuanDenseLayerWithBN(n, n_units=10, name='quan_dense_with_bn')
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

        cls.data["train_network"]["params"] = net_train.all_params

        cls.data["train_network"]["n_params"] = net_train.count_params()

        print(net_train.count_params())

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
