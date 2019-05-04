#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


def model(x, is_train=True, reuse=False, name_scope="env1"):
    with tf.variable_scope(name_scope, reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.TimeDistributedLayer(
            net, layer_class=tl.layers.DenseLayer, args={
                'n_units': 50,
                'name': 'dense'
            }, name='time_dense'
        )
    return net


class Layer_Time_Distributed_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        batch_size = 32
        timestep = 20
        input_dim = 100

        cls.x = tf.placeholder(dtype=tf.float32, shape=[batch_size, timestep, input_dim], name="encode_seqs")
        net = model(cls.x, is_train=True, reuse=False)

        cls.net_shape = net.outputs.get_shape().as_list()
        cls.n_params = net.count_params()
        net.print_params(False)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net_shape(self):
        self.assertEqual(self.net_shape, [32, 20, 50])

    def test_net_n_params(self):
        self.assertEqual(self.n_params, 5050)

    def test_reuse(self):

        with self.assertNotRaises(Exception):
            model(self.x, is_train=True, reuse=False, name_scope="env2")
            model(self.x, is_train=False, reuse=True, name_scope="env2")

        with self.assertRaises(Exception):
            model(self.x, is_train=True, reuse=False)  # Already defined model with the same var_scope


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
