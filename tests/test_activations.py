#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Test_Leaky_ReLUs(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.alpha = 0.2

        cls.input_var = tf.Variable(initial_value=0, dtype=tf.float32, name="Input_Var")

        cls.lrelu_out = tl.act.leaky_relu(cls.input_var, alpha=cls.alpha)  # Deprecated
        cls.lrelu6_out = tl.act.leaky_relu6(cls.input_var, alpha=cls.alpha)
        cls.ltrelu6_out = tl.act.leaky_twice_relu6(cls.input_var, alpha_low=cls.alpha, alpha_high=cls.alpha)

        cls.sess = tf.Session()
        cls.sess.run(tf.global_variables_initializer())

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_lrelu(self):
        for i in range(-5, 15):

            if i > 0:
                good_output = i
            else:
                good_output = self.alpha * i

            computed_output = self.sess.run(self.lrelu_out, feed_dict={self.input_var: i})

            self.assertAlmostEqual(computed_output, good_output, places=5)

    def test_lrelu6(self):
        for i in range(-5, 15):

            if i < 0:
                good_output = self.alpha * i
            else:
                good_output = min(6, i)

            computed_output = self.sess.run(self.lrelu6_out, feed_dict={self.input_var: i})

            self.assertAlmostEqual(computed_output, good_output, places=5)

    def test_ltrelu6(self):
        for i in range(-5, 15):

            if i < 0:
                good_output = self.alpha * i
            elif i < 6:
                good_output = i
            else:
                good_output = 6 + (self.alpha * (i - 6))

            computed_output = self.sess.run(self.ltrelu6_out, feed_dict={self.input_var: i})

            self.assertAlmostEqual(computed_output, good_output, places=5)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
