#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *

from tests.utils import CustomTestCase


class Layer_Stack_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("-" * 20, "Layer_Stack_Test", "-" * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]

        cls.ni = Input(cls.inputs_shape, name='input_layer')
        a = Dense(n_units=5)(cls.ni)
        b = Dense(n_units=5)(cls.ni)
        cls.layer1 = Stack(axis=1)
        cls.n1 = cls.layer1([a, b])
        cls.M = Model(inputs=cls.ni, outputs=cls.n1)

        cls.inputs = tf.random.uniform(cls.inputs_shape)
        cls.n2 = cls.M(cls.inputs, is_train=True)

        print(cls.layer1)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        self.assertEqual(self.n1.shape, (4, 2, 5))

    def test_layer_n2(self):
        self.assertEqual(self.n2.shape, (4, 2, 5))


class Layer_UnStack_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("-" * 20, "Layer_UnStack_Test", "-" * 20)
        cls.batch_size = 4
        cls.inputs_shape = [cls.batch_size, 10]

        cls.ni = Input(cls.inputs_shape, name='input_layer')
        a = Dense(n_units=5)(cls.ni)
        cls.layer1 = UnStack(axis=1)  # unstack in channel axis
        cls.n1 = cls.layer1(a)
        cls.M = Model(inputs=cls.ni, outputs=cls.n1)

        cls.inputs = tf.random.uniform(cls.inputs_shape)
        cls.n2 = cls.M(cls.inputs, is_train=True)

        print(cls.layer1)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_layer_n1(self):
        self.assertEqual(len(self.n1), 5)
        self.assertEqual(self.n1[0].shape, (self.batch_size, ))

    def test_layer_n2(self):
        self.assertEqual(len(self.n2), 5)
        self.assertEqual(self.n1[0].shape, (self.batch_size, ))


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
