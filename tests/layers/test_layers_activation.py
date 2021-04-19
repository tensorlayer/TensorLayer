#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Activation_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        self.inputs = tl.layers.Input([10, 5])

    @classmethod
    def tearDownClass(self):
        pass

    def test_prelu_1(self):
        prelulayer = tl.layers.PRelu(channel_shared=True)
        class prelu_model(tl.layers.Module):
            def __init__(self):
                super(prelu_model, self).__init__()
                self.prelu = prelulayer

            def forward(self, inputs):
                return self.prelu(inputs)
        net = prelu_model()

        self.assertTrue(tl.get_tensor_shape(net(self.inputs)), [10, 5])

    def test_prelu_2(self):
        prelulayer = tl.layers.PRelu(in_channels=5)
        prelu = prelulayer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(prelu), [10, 5])

    def test_prelu6_1(self):
        prelu6layer = tl.layers.PRelu6(in_channels=5)
        prelu6 = prelu6layer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(prelu6), [10, 5])


    def test_prelu6_2(self):
        prelu6layer = tl.layers.PRelu6(channel_shared=True)

        class prelu6_model(tl.layers.Module):
            def __init__(self):
                super(prelu6_model, self).__init__()
                self.prelu = prelu6layer

            def forward(self, inputs):
                return self.prelu(inputs)

        net = prelu6_model()

        self.assertTrue(tl.get_tensor_shape(net(self.inputs)), [10, 5])

    def test_ptrelu6_1(self):
        ptrelu6layer = tl.layers.PTRelu6(channel_shared=True)
        ptrelu6 = ptrelu6layer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(ptrelu6), [10, 5])

    def test_ptrelu6_2(self):
        ptrelu6layer = tl.layers.PTRelu6(in_channels=5)

        class ptrelu6_model(tl.layers.Module):
            def __init__(self):
                super(ptrelu6_model, self).__init__()
                self.prelu = ptrelu6layer

            def forward(self, inputs):
                return self.prelu(inputs)

        net = ptrelu6_model()

        self.assertTrue(tl.get_tensor_shape(net(self.inputs)), [10, 5])

    def test_lrelu(self):
        lrelulayer = tl.layers.LeakyReLU(alpha=0.5)
        lrelu = lrelulayer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(lrelu), [5, 10])

    def test_lrelu6(self):
        lrelu6layer = tl.layers.LeakyReLU6(alpha=0.5)
        lrelu6 = lrelu6layer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(lrelu6), [5, 10])

    def test_ltrelu6(self):
        ltrelu6layer = tl.layers.LeakyTwiceRelu6()
        ltrelu6 = ltrelu6layer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(ltrelu6), [5, 10])

    def test_swish(self):
        swishlayer = tl.layers.Swish()
        swish = swishlayer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(swish), [5, 10])

    def test_hardtanh(self):
        hardtanhlayer = tl.layers.HardTanh()
        hardtanh = hardtanhlayer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(hardtanh), [5, 10])

    def test_mish(self):
        mishlayer = tl.layers.Mish()
        mish = mishlayer(self.inputs)

        self.assertTrue(tl.get_tensor_shape(mish), [5, 10])


if __name__ == '__main__':

    unittest.main()
