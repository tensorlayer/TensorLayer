#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Extend_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_expand_dims(self):
        x = tl.layers.Input([8, 3])
        expandlayer = tl.layers.ExpandDims(axis=-1)
        y = expandlayer(x)
        self.assertEqual(tl.get_tensor_shape(y), [8, 3, 1])

    def test_tile(self):
        x = tl.layers.Input([8, 3])
        tilelayer = tl.layers.Tile(multiples=[2, 3])
        y = tilelayer(x)
        self.assertEqual(tl.get_tensor_shape(y), [16, 9])


if __name__ == '__main__':

    unittest.main()
