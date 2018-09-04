#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


def model(x, is_train, reuse):
    with tf.variable_scope("STN", reuse=reuse):
        nin = tl.layers.InputLayer(x, name='in')
        ## 1. Localisation network
        # use MLP as the localisation net
        nt = tl.layers.FlattenLayer(nin, name='flatten')
        nt = tl.layers.DenseLayer(nt, n_units=20, act=tf.nn.tanh, name='dense1')
        nt = tl.layers.DropoutLayer(nt, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        # you can also use CNN instead for MLP as the localisation net
        # nt = Conv2d(nin, 16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc1')
        # nt = Conv2d(nt, 8, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME', name='tc2')
        ## 2. Spatial transformer module (sampler)
        n = tl.layers.SpatialTransformer2dAffineLayer(nin, theta_layer=nt, out_size=(40, 40), name='spatial')
        s = n
        ## 3. Classifier
        n = tl.layers.Conv2d(
            n, n_filter=16, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, padding='SAME', name='conv1'
        )

        n = tl.layers.Conv2d(
            n, n_filter=16, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, padding='SAME', name='conv2'
        )
        n = tl.layers.FlattenLayer(n, name='flatten2')
        n = tl.layers.DenseLayer(n, n_units=1024, act=tf.nn.relu, name='out1')
        n = tl.layers.DenseLayer(n, n_units=10, name='out2')
    return n, s


class Layer_Spatial_Transformer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        net, s = model(cls.x, is_train=True, reuse=False)

        net.print_layers()
        net.print_params(False)

        cls.s_shape = s.outputs.get_shape().as_list()
        cls.net_layers = net.all_layers
        cls.net_params = net.all_params
        cls.net_n_params = net.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_reuse(self):

        with self.assertNotRaises(Exception):
            _, _ = model(self.x, is_train=True, reuse=True)

    def test_net_shape(self):
        self.assertEqual(self.s_shape[1:], [40, 40, 1])

    def test_net_layers(self):
        self.assertEqual(len(self.net_layers), 10)

    def test_net_params(self):
        self.assertEqual(len(self.net_params), 12)

    def test_net_n_params(self):
        self.assertEqual(self.net_n_params, 1667980)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
