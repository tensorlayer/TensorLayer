#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Shape_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = tl.layers.Input(shape=(8, 4, 3), init=tl.initializers.random_normal())
        cls.imgdata = tl.layers.Input(shape=(2, 16, 16, 8), init=tl.initializers.random_normal())

    @classmethod
    def tearDownClass(cls):
        pass

    def test_flatten(self):

        class CustomizeModel(tl.layers.Module):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.flatten = tl.layers.Flatten()

            def forward(self, x):
                return self.flatten(x)

        model = CustomizeModel()
        print(model.flatten)
        model.set_train()
        out = model(self.data)
        self.assertEqual(out.get_shape().as_list(), [8, 12])

    def test_reshape(self):

        class CustomizeModel(tl.layers.Module):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.reshape1 = tl.layers.Reshape(shape=(8, 12))
                self.reshape2 = tl.layers.Reshape(shape=(-1, 12))
                self.reshape3 = tl.layers.Reshape(shape=())

            def forward(self, x):
                return self.reshape1(x), self.reshape2(x), self.reshape3(x[0][0][0])

        model = CustomizeModel()
        print(model.reshape1)
        print(model.reshape2)
        print(model.reshape3)
        model.set_train()
        out1, out2, out3 = model(self.data)
        self.assertEqual(out1.get_shape().as_list(), [8, 12])
        self.assertEqual(out2.get_shape().as_list(), [8, 12])
        self.assertEqual(out3.get_shape().as_list(), [])

    def test_transpose(self):

        class CustomizeModel(tl.layers.Module):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.transpose1 = tl.layers.Transpose()
                self.transpose2 = tl.layers.Transpose([2, 1, 0])
                self.transpose3 = tl.layers.Transpose([0, 2, 1])
                self.transpose4 = tl.layers.Transpose(conjugate=True)

            def forward(self, x):
                return self.transpose1(x), self.transpose2(x), self.transpose3(x), self.transpose4(x)

        real = tl.layers.Input(shape=(8, 4, 3), init=tl.initializers.random_normal())
        comp = tl.layers.Input(shape=(8, 4, 3), init=tl.initializers.random_normal())
        import tensorflow as tf
        complex_data = tf.dtypes.complex(real, comp)
        model = CustomizeModel()
        print(model.transpose1)
        print(model.transpose2)
        print(model.transpose3)
        print(model.transpose4)
        model.set_train()
        out1, out2, out3, out4 = model(self.data)
        self.assertEqual(out1.get_shape().as_list(), [3, 4, 8])
        self.assertEqual(out2.get_shape().as_list(), [3, 4, 8])
        self.assertEqual(out3.get_shape().as_list(), [8, 3, 4])
        self.assertEqual(out4.get_shape().as_list(), [3, 4, 8])
        self.assertTrue(np.array_equal(out1.numpy(), out4.numpy()))

        out1, out2, out3, out4 = model(complex_data)
        self.assertEqual(out1.get_shape().as_list(), [3, 4, 8])
        self.assertEqual(out2.get_shape().as_list(), [3, 4, 8])
        self.assertEqual(out3.get_shape().as_list(), [8, 3, 4])
        self.assertEqual(out4.get_shape().as_list(), [3, 4, 8])
        self.assertTrue(np.array_equal(np.conj(out1.numpy()), out4.numpy()))

    def test_shuffle(self):

        class CustomizeModel(tl.layers.Module):

            def __init__(self, x):
                super(CustomizeModel, self).__init__()
                self.shuffle = tl.layers.Shuffle(x)

            def forward(self, x):
                return self.shuffle(x)

        model = CustomizeModel(2)
        print(model.shuffle)
        model.set_train()
        out = model(self.imgdata)
        self.assertEqual(out.get_shape().as_list(), [2, 16, 16, 8])


if __name__ == '__main__':

    unittest.main()
