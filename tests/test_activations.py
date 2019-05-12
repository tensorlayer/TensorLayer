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

        cls.vmin = 0
        cls.vmax = 10

    @classmethod
    def tearDownClass(cls):
        pass

    def test_lrelu(self):
        for i in range(-5, 15):

            if i > 0:
                good_output = i
            else:
                good_output = self.alpha * i

            computed_output = tl.act.leaky_relu(float(i), alpha=self.alpha)

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)

        net = tl.layers.Input([10, 2])
        net = tl.layers.Dense(n_units=100, act=lambda x: tl.act.lrelu(x, 0.2), name='dense')(net)
        print(net)

    def test_lrelu6(self):
        for i in range(-5, 15):

            if i < 0:
                good_output = self.alpha * i
            else:
                good_output = min(6, i)

            computed_output = tl.act.leaky_relu6(float(i), alpha=self.alpha)

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)

        net = tl.layers.Input([10, 2])
        net = tl.layers.Dense(n_units=100, act=lambda x: tl.act.leaky_relu6(x, 0.2), name='dense')(net)
        print(net)

    def test_ltrelu6(self):
        for i in range(-5, 15):

            if i < 0:
                good_output = self.alpha * i
            elif i < 6:
                good_output = i
            else:
                good_output = 6 + (self.alpha * (i - 6))

            computed_output = tl.act.leaky_twice_relu6(float(i), alpha_low=self.alpha, alpha_high=self.alpha)

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)

        net = tl.layers.Input([10, 200])
        net = tl.layers.Dense(n_units=100, act=lambda x: tl.act.leaky_twice_relu6(x, 0.2, 0.2), name='dense')(net)
        print(net)

    def test_ramp(self):

        for i in range(-5, 15):

            if i < self.vmin:
                good_output = self.vmin
            elif i > self.vmax:
                good_output = self.vmax
            else:
                good_output = i

            computed_output = tl.act.ramp(float(i), v_min=self.vmin, v_max=self.vmax)

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)

    def test_sign(self):

        for i in range(-5, 15):

            if i < 0:
                good_output = -1
            elif i == 0:
                good_output = 0
            else:
                good_output = 1

            computed_output = tl.act.sign(float(i))

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)

    def test_swish(self):
        import numpy as np

        for i in range(-5, 15):

            good_output = i / (1 + np.math.exp(-i))

            computed_output = tl.act.swish(float(i))

            self.assertAlmostEqual(computed_output.numpy(), good_output, places=5)


if __name__ == '__main__':

    unittest.main()
