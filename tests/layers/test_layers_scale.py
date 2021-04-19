#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Scale_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_scale(self):

        class model(tl.layers.Module):
            def __init__(self):
                super(model, self).__init__()
                self.dense = tl.layers.Dense(n_units=10)
                self.scalelayer = tl.layers.Scale(init_scale=0.5)

            def forward(self, inputs):
                output1 = self.dense(inputs)
                output2 = self.scalelayer(output1)
                return output1, output2

        input = tl.layers.Input((8, 3), init=tl.initializers.random_normal())
        net = model()
        net.set_train()
        dout, fout = net(input)

        for i in range(len(dout)):
            for j in range(len(dout[i])):
                self.assertEqual(dout[i][j].numpy() * 0.5, fout[i][j].numpy())


if __name__ == '__main__':

    unittest.main()
