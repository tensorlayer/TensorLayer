#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Activation_Layer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = (10 + 10) * np.random.random(size=[10, 5]).astype(np.float32) - 10
        cls.data2 = (10 + 10) * np.random.random(size=[10, 10, 5]).astype(np.float32) - 10

    @classmethod
    def tearDownClass(cls):
        pass

    def test_prelu_1(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PRelu(channel_shared=True)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0:
                    gt[i][j] = self.data[i][j]
                else:
                    gt[i][j] = prelulayer.alpha_var_constrained.numpy() * self.data[i][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_prelu_2(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PRelu(in_channels=5)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0:
                    gt[i][j] = self.data[i][j]
                else:
                    gt[i][j] = prelulayer.alpha_var_constrained.numpy()[j] * self.data[i][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_prelu_3(self):
        inputs = tl.layers.Input([10, 10, 5])
        prelulayer = tl.layers.PRelu(in_channels=5)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data2, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data2.shape)
        for i in range(len(gt)):
            for k in range(len(gt[i])):
                for j in range(len(gt[i][k])):
                    if self.data2[i][k][j] >= 0:
                        gt[i][k][j] = self.data2[i][k][j]
                    else:
                        gt[i][k][j] = prelulayer.alpha_var_constrained.numpy()[j] * self.data2[i][k][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_prelu6_1(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PRelu6(channel_shared=True)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0 and self.data[i][j] <= 6:
                    gt[i][j] = self.data[i][j]
                elif self.data[i][j] > 6:
                    gt[i][j] = 6
                else:
                    gt[i][j] = prelulayer.alpha_var_constrained.numpy() * self.data[i][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_prelu6_2(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PRelu6(in_channels=5)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0 and self.data[i][j] <= 6:
                    gt[i][j] = self.data[i][j]
                elif self.data[i][j] > 6:
                    gt[i][j] = 6
                else:
                    gt[i][j] = prelulayer.alpha_var_constrained.numpy()[j] * self.data[i][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_prelu6_3(self):
        inputs = tl.layers.Input([10, 10, 5])
        prelulayer = tl.layers.PRelu6(in_channels=5)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data2, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data2.shape)
        for i in range(len(gt)):
            for k in range(len(gt[i])):
                for j in range(len(gt[i][k])):
                    if self.data2[i][k][j] >= 0 and self.data2[i][k][j] <= 6:
                        gt[i][k][j] = self.data2[i][k][j]
                    elif self.data2[i][k][j] > 6:
                        gt[i][k][j] = 6
                    else:
                        gt[i][k][j] = prelulayer.alpha_var_constrained.numpy()[j] * self.data2[i][k][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_ptrelu6_1(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PTRelu6(channel_shared=True)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0 and self.data[i][j] <= 6:
                    gt[i][j] = self.data[i][j]
                elif self.data[i][j] > 6:
                    gt[i][j] = 6 + prelulayer.alpha_high_constrained.numpy() * (self.data[i][j] - 6)
                else:
                    gt[i][j] = prelulayer.alpha_low_constrained.numpy() * self.data[i][j]

        self.assertTrue(np.array_equal(out.numpy(), gt))

    def test_ptrelu6_2(self):
        inputs = tl.layers.Input([10, 5])
        prelulayer = tl.layers.PTRelu6(in_channels=5)
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data.shape)
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                if self.data[i][j] >= 0 and self.data[i][j] <= 6:
                    gt[i][j] = self.data[i][j]
                elif self.data[i][j] > 6:
                    gt[i][j] = 6 + prelulayer.alpha_high_constrained.numpy()[j] * (self.data[i][j] - 6)
                else:
                    gt[i][j] = prelulayer.alpha_low_constrained.numpy()[j] * self.data[i][j]

        self.assertTrue(np.allclose(out.numpy(), gt))

    def test_ptrelu6_3(self):
        inputs = tl.layers.Input([3, 2, 5])
        prelulayer = tl.layers.PTRelu6()
        prelu = prelulayer(inputs)
        model = tl.models.Model(inputs=inputs, outputs=prelu)
        out = model(self.data2, is_train=True)

        print(prelulayer)

        gt = np.zeros(shape=self.data2.shape)
        for i in range(len(gt)):
            for k in range(len(gt[i])):
                for j in range(len(gt[i][k])):
                    if self.data2[i][k][j] >= 0 and self.data2[i][k][j] <= 6:
                        gt[i][k][j] = self.data2[i][k][j]
                    elif self.data2[i][k][j] > 6:
                        gt[i][k][j] = 6 + prelulayer.alpha_high_constrained.numpy()[j] * (self.data2[i][k][j] - 6)
                    else:
                        gt[i][k][j] = prelulayer.alpha_low_constrained.numpy()[j] * self.data2[i][k][j]

        self.assertTrue(np.allclose(out.numpy(), gt))


if __name__ == '__main__':

    unittest.main()
