#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tests.utils import CustomTestCase


class Test_Leaky_ReLUs(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.ni = tl.layers.Input(shape=[16, 10])
        cls.w_shape = (10, 5)
        cls.eps = 0.0

    @classmethod
    def tearDownClass(cls):
        pass

    def init_dense(self, w_init):
        return tl.layers.Dense(n_units=self.w_shape[1], in_channels=self.w_shape[0], W_init=w_init)

    def test_zeros(self):
        dense = self.init_dense(tl.initializers.zeros())
        self.assertEqual(np.sum(dense.all_weights[0].numpy() - np.zeros(shape=self.w_shape)), self.eps)
        nn = dense(self.ni)

    def test_ones(self):
        dense = self.init_dense(tl.initializers.ones())
        self.assertEqual(np.sum(dense.all_weights[0].numpy() - np.ones(shape=self.w_shape)), self.eps)
        nn = dense(self.ni)

    def test_constant(self):
        dense = self.init_dense(tl.initializers.constant(value=5.0))
        self.assertEqual(np.sum(dense.all_weights[0].numpy() - np.ones(shape=self.w_shape) * 5.0), self.eps)
        nn = dense(self.ni)

        # test with numpy arr
        arr = np.random.uniform(size=self.w_shape).astype(np.float32)
        dense = self.init_dense(tl.initializers.constant(value=arr))
        self.assertEqual(np.sum(dense.all_weights[0].numpy() - arr), self.eps)
        nn = dense(self.ni)

    def test_RandomUniform(self):
        dense = self.init_dense(tl.initializers.random_uniform(minval=-0.1, maxval=0.1, seed=1234))
        print(dense.all_weights[0].numpy())
        nn = dense(self.ni)

    def test_RandomNormal(self):
        dense = self.init_dense(tl.initializers.random_normal(mean=0.0, stddev=0.1))
        print(dense.all_weights[0].numpy())
        nn = dense(self.ni)

    def test_TruncatedNormal(self):
        dense = self.init_dense(tl.initializers.truncated_normal(mean=0.0, stddev=0.1))
        print(dense.all_weights[0].numpy())
        nn = dense(self.ni)

    def test_deconv2d_bilinear_upsampling_initializer(self):
        rescale_factor = 2
        imsize = 128
        num_channels = 3
        num_in_channels = 3
        num_out_channels = 3
        filter_shape = (5, 5, num_out_channels, num_in_channels)
        ni = tl.layers.Input(shape=(1, imsize, imsize, num_channels))
        bilinear_init = tl.initializers.deconv2d_bilinear_upsampling_initializer(shape=filter_shape)
        deconv_layer = tl.layers.DeConv2dLayer(
            shape=filter_shape, outputs_shape=(1, imsize * rescale_factor, imsize * rescale_factor, num_out_channels),
            strides=(1, rescale_factor, rescale_factor, 1), W_init=bilinear_init, padding='SAME', act=None,
            name='g/h1/decon2d'
        )
        nn = deconv_layer(ni)

    def test_config(self):
        init = tl.initializers.constant(value=5.0)
        new_init = tl.initializers.Constant.from_config(init.get_config())


if __name__ == '__main__':

    unittest.main()
