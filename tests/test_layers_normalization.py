#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.Conv2d(n, n_filter=80, name='conv2d_1')
        n = tl.layers.BatchNormLayer(n, is_train=is_train, name='norm_batch')
        n = tl.layers.Conv2d(n, n_filter=80, name='conv2d_2')
        n = tl.layers.LocalResponseNormLayer(n, name='norm_local')
        n = tl.layers.LayerNormLayer(n, reuse=reuse, name='norm_layer')
        n = tl.layers.InstanceNormLayer(n, name='norm_instance')
    return n


class Layer_Normalization_Test(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_all_layers(self):
        self.assertEqual(len(self.data["train_network"]["layers"]), 6)
        self.assertEqual(len(self.data["eval_network"]["layers"]), 6)

    def test_all_params(self):
        self.assertEqual(len(self.data["train_network"]["params"]), 12)

    def test_n_params(self):
        self.assertEqual(self.data["train_network"]["n_params"], 60560)


if __name__ == '__main__':

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
