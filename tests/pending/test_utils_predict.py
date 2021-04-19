#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Util_Predict_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.x1 = tf.placeholder(tf.float32, [None, 5, 5, 3])
        cls.x2 = tf.placeholder(tf.float32, [8, 5, 5, 3])
        cls.X1 = np.ones([127, 5, 5, 3])
        cls.X2 = np.ones([7, 5, 5, 3])
        cls.batch_size = 8

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_case1(self):
        with self.assertNotRaises(Exception):
            with tf.Session() as sess:
                n = tl.layers.InputLayer(self.x1)
                y = n.outputs
                y_op = tf.nn.softmax(y)
                tl.utils.predict(sess, n, self.X1, self.x1, y_op, batch_size=self.batch_size)
                sess.close()

    def test_case2(self):
        with self.assertRaises(Exception):
            with tf.Session() as sess:
                n = tl.layers.InputLayer(self.x2)
                y = n.outputs
                y_op = tf.nn.softmax(y)
                tl.utils.predict(sess, n, self.X2, self.x2, y_op, batch_size=self.batch_size)
                sess.close()


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
