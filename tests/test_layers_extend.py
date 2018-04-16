#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Extend_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        x = tf.placeholder(tf.float32, (None, 100))
        n = tl.layers.InputLayer(x, name='in')
        n = tl.layers.DenseLayer(n, n_units=100, name='d1')
        n = tl.layers.DenseLayer(n, n_units=100, name='d2')

        ## 1D

        n = tl.layers.ExpandDimsLayer(n, axis=2)
        cls.shape_1 = n.outputs.get_shape().as_list()

        n = tl.layers.TileLayer(n, multiples=[-1, 1, 3])
        cls.shape_2 = n.outputs.get_shape().as_list()

        n.print_layers()
        n.print_params(False)

        cls.layers = n.all_layers
        cls.params = n.all_params

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_shape_1(self):
        assert (self.shape_1[-1] == 1)

    def test_shape_2(self):
        assert (self.shape_2[-1] == 3)

    def test_layers(self):
        assert (len(self.layers) == 4)

    def test_params(self):
        assert (len(self.params) == 4)


if __name__ == '__main__':

    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
