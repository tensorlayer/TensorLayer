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


def basic_static_model(include_top=True):
    ni = Input((None, 24, 24, 3))
    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name="conv1")(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)

    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name="conv2")(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(100, act=None, name="dense1")(nn)
    if include_top is True:
        nn = Dense(10, act=None, name="dense2")(nn)
    M = Model(inputs=ni, outputs=nn)
    return M


class basic_dynamic_model(Model):

    def __init__(self, include_top=True):
        super(basic_dynamic_model, self).__init__()
        self.include_top = include_top
        self.conv1 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=3, name="conv1")
        self.pool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=16, name="conv2")
        self.pool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(100, act=None, in_channels=576, name="dense1")
        if include_top is True:
            self.dense2 = Dense(10, act=None, in_channels=100, name="dense2")

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if self.include_top:
            x = self.dense2(x)
        return x


class Nested_VGG(Model):

    def __init__(self):
        super(Nested_VGG, self).__init__()
        self.vgg1 = tl.models.vgg16()
        self.vgg2 = tl.models.vgg16()

    def forward(self, x):
        pass


class Model_Save_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.static_basic = basic_static_model()
        cls.dynamic_basic = basic_dynamic_model()
        cls.static_basic_skip = basic_static_model(include_top=False)
        cls.dynamic_basic_skip = basic_dynamic_model(include_top=False)

        print([l.name for l in cls.dynamic_basic.all_layers])
        print([l.name for l in cls.dynamic_basic_skip.all_layers])
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def normal_save(self, model_basic):
        # Default save
        model_basic.save_weights('./model_basic.none')

        # hdf5
        print('testing hdf5 saving...')
        modify_val = np.zeros_like(model_basic.all_weights[-2].numpy())
        ori_val = model_basic.all_weights[-2].numpy()
        model_basic.save_weights("./model_basic.h5")
        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.h5")
        self.assertLess(np.max(np.abs(ori_val - model_basic.all_weights[-2].numpy())), 1e-7)

        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.h5", format="hdf5")
        self.assertLess(np.max(np.abs(ori_val - model_basic.all_weights[-2].numpy())), 1e-7)

        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.h5", format="hdf5", in_order=False)
        self.assertLess(np.max(np.abs(ori_val - model_basic.all_weights[-2].numpy())), 1e-7)

        # npz
        print('testing npz saving...')
        model_basic.save_weights("./model_basic.npz", format='npz')
        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.npz")

        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.npz", format='npz')
        model_basic.save_weights("./model_basic.npz")
        self.assertLess(np.max(np.abs(ori_val - model_basic.all_weights[-2].numpy())), 1e-7)

        # npz_dict
        print('testing npz_dict saving...')
        model_basic.save_weights("./model_basic.npz", format='npz_dict')
        model_basic.all_weights[-2].assign(modify_val)
        model_basic.load_weights("./model_basic.npz", format='npz_dict')
        self.assertLess(np.max(np.abs(ori_val - model_basic.all_weights[-2].numpy())), 1e-7)

        # ckpt
        try:
            model_basic.save_weights('./model_basic.ckpt', format='ckpt')
        except Exception as e:
            self.assertIsInstance(e, NotImplementedError)

        # other cases
        try:
            model_basic.save_weights('./model_basic.xyz', format='xyz')
        except Exception as e:
            self.assertIsInstance(e, ValueError)
        try:
            model_basic.load_weights('./model_basic.xyz', format='xyz')
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError)
        try:
            model_basic.load_weights('./model_basic.h5', format='xyz')
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_normal_save(self):
        print('-' * 20, 'test save weights', '-' * 20)

        self.normal_save(self.static_basic)
        self.normal_save(self.dynamic_basic)

        print('testing save dynamic and load static...')
        try:
            self.dynamic_basic.save_weights("./model_basic.h5")
            self.static_basic.load_weights("./model_basic.h5", in_order=False)
        except Exception as e:
            print(e)

    def test_skip(self):
        print('-' * 20, 'test skip save/load', '-' * 20)

        print("testing dynamic skip load...")
        self.dynamic_basic.save_weights("./model_basic.h5")
        ori_weights = self.dynamic_basic_skip.all_weights
        ori_val = ori_weights[1].numpy()
        modify_val = np.zeros_like(ori_val)
        self.dynamic_basic_skip.all_weights[1].assign(modify_val)
        self.dynamic_basic_skip.load_weights("./model_basic.h5", skip=True)
        self.assertLess(np.max(np.abs(ori_val - self.dynamic_basic_skip.all_weights[1].numpy())), 1e-7)

        try:
            self.dynamic_basic_skip.load_weights("./model_basic.h5", in_order=False, skip=False)
        except Exception as e:
            print(e)

        print("testing static skip load...")
        self.static_basic.save_weights("./model_basic.h5")
        ori_weights = self.static_basic_skip.all_weights
        ori_val = ori_weights[1].numpy()
        modify_val = np.zeros_like(ori_val)
        self.static_basic_skip.all_weights[1].assign(modify_val)
        self.static_basic_skip.load_weights("./model_basic.h5", skip=True)
        self.assertLess(np.max(np.abs(ori_val - self.static_basic_skip.all_weights[1].numpy())), 1e-7)

        try:
            self.static_basic_skip.load_weights("./model_basic.h5", in_order=False, skip=False)
        except Exception as e:
            print(e)

    def test_nested_vgg(self):
        print('-' * 20, 'test nested vgg', '-' * 20)
        nested_vgg = Nested_VGG()
        print([l.name for l in nested_vgg.all_layers])
        nested_vgg.save_weights("nested_vgg.h5")

        # modify vgg1 weight val
        tar_weight1 = nested_vgg.vgg1.layers[0].all_weights[0]
        print(tar_weight1.name)
        ori_val1 = tar_weight1.numpy()
        modify_val1 = np.zeros_like(ori_val1)
        tar_weight1.assign(modify_val1)
        # modify vgg2 weight val
        tar_weight2 = nested_vgg.vgg2.layers[1].all_weights[0]
        print(tar_weight2.name)
        ori_val2 = tar_weight2.numpy()
        modify_val2 = np.zeros_like(ori_val2)
        tar_weight2.assign(modify_val2)

        nested_vgg.load_weights("nested_vgg.h5")

        self.assertLess(np.max(np.abs(ori_val1 - tar_weight1.numpy())), 1e-7)
        self.assertLess(np.max(np.abs(ori_val2 - tar_weight2.numpy())), 1e-7)

    def test_double_nested_vgg(self):
        print('-' * 20, 'test_double_nested_vgg', '-' * 20)

        class mymodel(Model):

            def __init__(self):
                super(mymodel, self).__init__()
                self.inner = Nested_VGG()
                self.list = LayerList(
                    [
                        tl.layers.Dense(n_units=4, in_channels=10, name='dense1'),
                        tl.layers.Dense(n_units=3, in_channels=4, name='dense2')
                    ]
                )

            def forward(self, *inputs, **kwargs):
                pass

        net = mymodel()
        net.save_weights("double_nested.h5")
        print([x.name for x in net.all_layers])

        # modify vgg1 weight val
        tar_weight1 = net.inner.vgg1.layers[0].all_weights[0]
        ori_val1 = tar_weight1.numpy()
        modify_val1 = np.zeros_like(ori_val1)
        tar_weight1.assign(modify_val1)
        # modify vgg2 weight val
        tar_weight2 = net.inner.vgg2.layers[1].all_weights[0]
        ori_val2 = tar_weight2.numpy()
        modify_val2 = np.zeros_like(ori_val2)
        tar_weight2.assign(modify_val2)

        net.load_weights("double_nested.h5")
        self.assertLess(np.max(np.abs(ori_val1 - tar_weight1.numpy())), 1e-7)
        self.assertLess(np.max(np.abs(ori_val2 - tar_weight2.numpy())), 1e-7)

    def test_layerlist(self):
        print('-' * 20, 'test_layerlist', '-' * 20)

        # simple modellayer
        ni = tl.layers.Input([10, 4])
        nn = tl.layers.Dense(n_units=3, name='dense1')(ni)
        modellayer = tl.models.Model(inputs=ni, outputs=nn, name='modellayer').as_layer()

        # nested layerlist with modellayer
        inputs = tl.layers.Input([10, 5])
        layer1 = tl.layers.LayerList([tl.layers.Dense(n_units=4, name='dense1'), modellayer])(inputs)
        model = tl.models.Model(inputs=inputs, outputs=layer1, name='layerlistmodel')

        model.save_weights("layerlist.h5")
        tar_weight = model.get_layer(index=-1)[0].all_weights[0]
        print(tar_weight.name)
        ori_val = tar_weight.numpy()
        modify_val = np.zeros_like(ori_val)
        tar_weight.assign(modify_val)

        model.load_weights("layerlist.h5")
        self.assertLess(np.max(np.abs(ori_val - tar_weight.numpy())), 1e-7)

    def test_exceptions(self):
        print('-' * 20, 'test_exceptions', '-' * 20)
        try:
            ni = Input([4, 784])
            model = Model(inputs=ni, outputs=ni)
            model.save_weights('./empty_model.h5')
        except Exception as e:
            print(e)


if __name__ == '__main__':

    unittest.main()
