#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model
from tests.utils import CustomTestCase


class Laye_BatchNorm_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        x_0_input_shape = [None, 10]
        x_1_input_shape = [None, 100, 1]
        x_2_input_shape = [None, 100, 100, 3]
        x_3_input_shape = [None, 100, 100, 100, 3]
        batchsize = 2

        cls.x0 = tf.random.normal([batchsize] + x_0_input_shape[1:])
        cls.x1 = tf.random.normal([batchsize] + x_1_input_shape[1:])
        cls.x2 = tf.random.normal([batchsize] + x_2_input_shape[1:])
        cls.x3 = tf.random.normal([batchsize] + x_3_input_shape[1:])

        ## Base
        ni_1 = Input(x_1_input_shape, name='test_ni1')
        nn_1 = Conv1d(n_filter=32, filter_size=5, stride=2, name='test_conv1d')(ni_1)
        n1_b = BatchNorm(name='test_conv')(nn_1)
        cls.n1_b = n1_b
        cls.base_1d = Model(inputs=ni_1, outputs=n1_b, name='test_base_1d')

        ni_2 = Input(x_2_input_shape, name='test_ni2')
        nn_2 = Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='test_conv2d')(ni_2)
        n2_b = BatchNorm(name='test_bn2d')(nn_2)
        cls.n2_b = n2_b
        cls.base_2d = Model(inputs=ni_2, outputs=n2_b, name='test_base_2d')

        ni_3 = Input(x_3_input_shape, name='test_ni2')
        nn_3 = Conv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), name='test_conv3d')(ni_3)
        n3_b = BatchNorm(name='test_bn3d')(nn_3)
        cls.n3_b = n3_b
        cls.base_3d = Model(inputs=ni_3, outputs=n3_b, name='test_base_3d')

        class bn_0d_model(Model):

            def __init__(self):
                super(bn_0d_model, self).__init__()
                self.fc = Dense(32, in_channels=10)
                self.bn = BatchNorm(num_features=32, name='test_bn1d')

            def forward(self, x):
                x = self.bn(self.fc(x))
                return x

        dynamic_base = bn_0d_model()
        cls.n0_b = dynamic_base(cls.x0, is_train=True)

        ## 0D ========================================================================

        nin_0 = Input(x_0_input_shape, name='test_in1')

        n0 = Dense(32)(nin_0)
        n0 = BatchNorm1d(name='test_bn0d')(n0)

        cls.n0 = n0

        cls.static_0d = Model(inputs=nin_0, outputs=n0)

        class bn_0d_model(Model):

            def __init__(self):
                super(bn_0d_model, self).__init__(name='test_bn_0d_model')
                self.fc = Dense(32, in_channels=10)
                self.bn = BatchNorm1d(num_features=32, name='test_bn1d')

            def forward(self, x):
                x = self.bn(self.fc(x))
                return x

        cls.dynamic_0d = bn_0d_model()

        print("Printing BatchNorm0d")
        print(cls.static_0d)
        print(cls.dynamic_0d)

        ## 1D ========================================================================

        nin_1 = Input(x_1_input_shape, name='test_in1')

        n1 = Conv1d(n_filter=32, filter_size=5, stride=2, name='test_conv1d')(nin_1)
        n1 = BatchNorm1d(name='test_bn1d')(n1)

        cls.n1 = n1

        cls.static_1d = Model(inputs=nin_1, outputs=n1)

        class bn_1d_model(Model):

            def __init__(self):
                super(bn_1d_model, self).__init__(name='test_bn_1d_model')
                self.conv = Conv1d(n_filter=32, filter_size=5, stride=2, name='test_conv1d', in_channels=1)
                self.bn = BatchNorm1d(num_features=32, name='test_bn1d')

            def forward(self, x):
                x = self.bn(self.conv(x))
                return x

        cls.dynamic_1d = bn_1d_model()

        print("Printing BatchNorm1d")
        print(cls.static_1d)
        print(cls.dynamic_1d)

        ## 2D ========================================================================

        nin_2 = Input(x_2_input_shape, name='test_in2')

        n2 = Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='test_conv2d')(nin_2)
        n2 = BatchNorm2d(name='test_bn2d')(n2)

        cls.n2 = n2

        cls.static_2d = Model(inputs=nin_2, outputs=n2)

        class bn_2d_model(Model):

            def __init__(self):
                super(bn_2d_model, self).__init__(name='test_bn_2d_model')
                self.conv = Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='test_conv2d', in_channels=3)
                self.bn = BatchNorm2d(num_features=32, name='test_bn2d')

            def forward(self, x):
                x = self.bn(self.conv(x))
                return x

        cls.dynamic_2d = bn_2d_model()

        print("Printing BatchNorm1d")
        print(cls.static_2d)
        print(cls.dynamic_2d)

        ## 3D ========================================================================

        nin_3 = Input(x_3_input_shape, name='test_in3')

        n3 = Conv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), name='test_conv3d')(nin_3)
        n3 = BatchNorm3d(name='test_bn3d', act=tf.nn.relu)(n3)

        cls.n3 = n3

        cls.static_3d = Model(inputs=nin_3, outputs=n3)

        class bn_3d_model(Model):

            def __init__(self):
                super(bn_3d_model, self).__init__(name='test_bn_3d_model')
                self.conv = Conv3d(
                    n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2), name='test_conv3d', in_channels=3
                )
                self.bn = BatchNorm3d(num_features=32, name='test_bn3d')

            def forward(self, x):
                x = self.bn(self.conv(x))
                return x

        cls.dynamic_3d = bn_3d_model()

        print("Printing BatchNorm1d")
        print(cls.static_3d)
        print(cls.dynamic_3d)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_BatchNorm(self):
        self.assertEqual(self.n1_b.shape[1:], (50, 32))
        out = self.base_1d(self.x1, is_train=True)

        self.assertEqual(self.n2_b.shape[1:], (50, 50, 32))
        out = self.base_2d(self.x2, is_train=True)

        self.assertEqual(self.n3_b.shape[1:], (50, 50, 50, 32))
        out = self.base_3d(self.x3, is_train=True)

        self.assertEqual(self.n0_b.shape[1:], (32))
        print("test_BatchNorm OK")

    def test_BatchNorm0d(self):
        self.assertEqual(self.n0.shape[1:], (32))
        out = self.static_0d(self.x0, is_train=True)
        out = self.dynamic_0d(self.x0, is_train=True)

    def test_BatchNorm1d(self):
        self.assertEqual(self.n1.shape[1:], (50, 32))
        out = self.static_1d(self.x1, is_train=True)
        out = self.dynamic_1d(self.x1, is_train=True)

    def test_BatchNorm2d(self):
        self.assertEqual(self.n2.shape[1:], (50, 50, 32))
        out = self.static_2d(self.x2, is_train=True)
        out = self.dynamic_2d(self.x2, is_train=True)
        out = self.dynamic_2d(self.x2, is_train=False)

    def test_BatchNorm3d(self):
        self.assertEqual(self.n3.shape[1:], (50, 50, 50, 32))
        out = self.static_3d(self.x3, is_train=True)
        out = self.dynamic_3d(self.x3, is_train=True)

    def test_dataformat(self):
        bn1d = BatchNorm1d(data_format='channels_first', num_features=32)
        bn2d = BatchNorm2d(data_format='channels_first', num_features=32)
        bn3d = BatchNorm3d(data_format='channels_first', num_features=32)
        bn = BatchNorm(data_format='channels_first')

        try:
            bn_fail = BatchNorm1d(data_format='xyz', num_features=32)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

    def test_exception(self):
        try:
            bn = BatchNorm(num_features=32)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

        try:
            ni = Input([None, 100, 1], name='test_ni1')
            bn = BatchNorm(decay=1.5)(ni)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

    def test_input_shape(self):
        try:
            bn = BatchNorm1d(num_features=32)
            out = bn(self.x2)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)
        try:
            bn = BatchNorm2d(num_features=32)
            out = bn(self.x3)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)
        try:
            bn = BatchNorm3d(num_features=32)
            out = bn(self.x1)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
