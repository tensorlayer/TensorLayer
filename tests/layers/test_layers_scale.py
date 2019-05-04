#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
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
        inputs = tl.layers.Input([8, 3])
        dense = tl.layers.Dense(n_units=10)(inputs)
        scalelayer = tl.layers.Scale(init_scale=0.5)
        outputs = scalelayer(dense)
        model = tl.models.Model(inputs=inputs, outputs=[dense, outputs])

        print(scalelayer)

        data = np.random.random(size=[8, 3]).astype(np.float32)
        dout, fout = model(data, is_train=True)

        for i in range(len(dout)):
            for j in range(len(dout[i])):
                self.assertEqual(dout[i][j].numpy() * 0.5, fout[i][j].numpy())


if __name__ == '__main__':

    unittest.main()
