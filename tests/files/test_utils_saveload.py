#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *

from tests.utils import CustomTestCase


def basic_static_model():
    ni = Input((None, 24, 24, 3))
    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name="conv1")(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)

    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name="conv2")(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(100, act=None, name="dense1")(nn)
    nn = Dense(10, act=None, name="dense2")(nn)
    M = Model(inputs=ni, outputs=nn, name='basic_static')
    return M


class basic_dynamic_model(Model):

    def __init__(self):
        super(basic_dynamic_model, self).__init__(name="basic_dynamic")
        self.conv1 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=3, name="conv1")
        self.pool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=16, name="conv2")
        self.pool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(100, act=None, in_channels=576, name="dense1")
        self.dense2 = Dense(10, act=None, in_channels=100, name="dense2")

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class Model_Core_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.static_model = basic_static_model()
        cls.dynamic_model = basic_dynamic_model()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_hdf5(self):
        modify_val = np.zeros_like(self.static_model.all_weights[-2].numpy())
        ori_val = self.static_model.all_weights[-2].numpy()
        tl.files.save_weights_to_hdf5("./model_basic.h5", self.static_model)

        self.static_model.all_weights[-2].assign(modify_val)
        tl.files.load_hdf5_to_weights_in_order("./model_basic.h5", self.static_model)
        self.assertLess(np.max(np.abs(ori_val - self.static_model.all_weights[-2].numpy())), 1e-7)

        self.static_model.all_weights[-2].assign(modify_val)
        tl.files.load_hdf5_to_weights("./model_basic.h5", self.static_model)
        self.assertLess(np.max(np.abs(ori_val - self.static_model.all_weights[-2].numpy())), 1e-7)

        ori_weights = self.static_model._all_weights
        self.static_model._all_weights = self.static_model._all_weights[1:]
        self.static_model.all_weights[-2].assign(modify_val)
        tl.files.load_hdf5_to_weights("./model_basic.h5", self.static_model, skip=True)
        self.assertLess(np.max(np.abs(ori_val - self.static_model.all_weights[-2].numpy())), 1e-7)
        self.static_model._all_weights = ori_weights

    def test_npz(self):
        modify_val = np.zeros_like(self.dynamic_model.all_weights[-2].numpy())
        ori_val = self.dynamic_model.all_weights[-2].numpy()
        tl.files.save_npz(self.dynamic_model.all_weights, "./model_basic.npz")

        self.dynamic_model.all_weights[-2].assign(modify_val)
        tl.files.load_and_assign_npz("./model_basic.npz", self.dynamic_model)
        self.assertLess(np.max(np.abs(ori_val - self.dynamic_model.all_weights[-2].numpy())), 1e-7)

    def test_npz_dict(self):
        modify_val = np.zeros_like(self.dynamic_model.all_weights[-2].numpy())
        ori_val = self.dynamic_model.all_weights[-2].numpy()
        tl.files.save_npz_dict(self.dynamic_model.all_weights, "./model_basic.npz")

        self.dynamic_model.all_weights[-2].assign(modify_val)
        tl.files.load_and_assign_npz_dict("./model_basic.npz", self.dynamic_model)
        self.assertLess(np.max(np.abs(ori_val - self.dynamic_model.all_weights[-2].numpy())), 1e-7)

        ori_weights = self.dynamic_model._all_weights
        self.dynamic_model._all_weights = self.static_model._all_weights[1:]
        self.dynamic_model.all_weights[-2].assign(modify_val)
        tl.files.load_and_assign_npz_dict("./model_basic.npz", self.dynamic_model, skip=True)
        self.assertLess(np.max(np.abs(ori_val - self.dynamic_model.all_weights[-2].numpy())), 1e-7)
        self.dynamic_model._all_weights = ori_weights


if __name__ == '__main__':

    unittest.main()
