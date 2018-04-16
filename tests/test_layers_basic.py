#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Basic_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, [None, 100])
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.DenseLayer(n, n_units=80, name='d1')
        n = tl.layers.DenseLayer(n, n_units=80, name='d2')

        n.print_layers()
        n.print_params(False)

        n2 = n[:, :30]
        n2.print_layers()

        cls.n_params = n.count_params()
        cls.shape_n = n.outputs.get_shape().as_list()
        cls.shape_n2 = n2.outputs.get_shape().as_list()
        cls.all_layers = n.all_layers
        cls.all_params = n.all_params

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_n_params(self):
        assert (self.n_params == 14560)

    def test_shape_n(self):
        assert (self.shape_n[-1] == 80)

    def test_all_layers(self):
        assert (len(self.all_layers) == 2)

    def test_all_params(self):
        assert (len(self.all_params) == 4)

    def test_shape_n2(self):
        assert (self.shape_n2[-1] == 30)


if __name__ == '__main__':

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
