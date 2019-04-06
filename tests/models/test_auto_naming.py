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


def basic_static_model(name=None, conv1_name="conv1", conv2_name="conv2"):
    ni = Input((None, 24, 24, 3))
    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name=conv1_name)(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)

    nn = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, name=conv2_name)(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    M = Model(inputs=ni, outputs=nn, name=name)
    return M


def nested_static_model(name=None, inner_model_name=None):
    ni = Input((None, 24, 24, 3))
    nn = ModelLayer(basic_static_model(inner_model_name))(ni)
    M = Model(inputs=ni, outputs=nn, name=name)
    return M


class basic_dynamic_model(Model):
    def __init__(self, name=None, conv1_name="conv1", conv2_name="conv2"):
        super(basic_dynamic_model, self).__init__(name=name)
        self.conv1 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=3, name=conv1_name)
        self.pool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(16, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, in_channels=16, name=conv2_name)
        self.pool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return x


class nested_dynamic_model(Model):
    def __init__(self, name=None, inner_model_name_1=None, inner_model_name_2=None):
        super(nested_dynamic_model, self).__init__(name=name)

        self.inner_model_1 = basic_dynamic_model(name=inner_model_name_1)
        self.inner_model_2 = basic_dynamic_model(name=inner_model_name_2)

    def forward(self, x):
        x = self.inner_model_1(x)
        x = self.inner_model_2(x)
        return x


class Auto_Naming_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dynamic_model_auto_naming(self):
        print('-' * 20, 'test_dynamic_model_auto_naming', '-' * 20)
        test_flag = True

        model_basic = basic_dynamic_model()
        model_basic_1 = basic_dynamic_model()
        model_basic_2 = basic_dynamic_model("basic_dynamic_model_2")
        model_basic_3 = basic_dynamic_model()
        model_basic_given_name = basic_dynamic_model("a_dynamic_model")

        self.assertEqual(model_basic.name, "basic_dynamic_model")
        self.assertEqual(model_basic.conv1.name, "conv1")
        self.assertEqual(model_basic_1.name, "basic_dynamic_model_1")
        self.assertEqual(model_basic_1.conv1.name, "conv1")
        self.assertEqual(model_basic_2.name, "basic_dynamic_model_2")
        self.assertEqual(model_basic_3.name, "basic_dynamic_model_3")
        self.assertEqual(model_basic_given_name.name, "a_dynamic_model")

        try:
            model_basic_given_repeat_name = basic_dynamic_model("basic_dynamic_model_1")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect repeat user given names")

        model_nested = nested_dynamic_model()
        model_nested_1 = nested_dynamic_model(inner_model_name_1="a_inner_dynamic_model")

        self.assertEqual(model_nested.name, "nested_dynamic_model")
        self.assertEqual(model_nested.inner_model_1.name, "basic_dynamic_model_4")
        self.assertEqual(model_nested.inner_model_2.name, "basic_dynamic_model_5")
        self.assertEqual(model_nested_1.name, "nested_dynamic_model_1")
        self.assertEqual(model_nested_1.inner_model_1.name, "a_inner_dynamic_model")
        self.assertEqual(model_nested_1.inner_model_2.name, "basic_dynamic_model_6")

        try:
            model_nested_given_repeat_name = nested_dynamic_model(inner_model_name_2="basic_dynamic_model_1")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect nested repeat user given names")

        try:
            model_nested_given_repeat_name_1 = nested_dynamic_model(name="basic_dynamic_model_5")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect nested repeat user given names")

    def test_static_model_auto_naming(self):
        print('-' * 20, 'test_static_model_auto_naming', '-' * 20)
        test_flag = True

        model_basic = basic_static_model()
        model_basic_1 = basic_static_model()
        model_basic_2 = basic_static_model("model_2")
        model_basic_3 = basic_static_model()
        model_basic_given_name = basic_static_model("a_static_model")

        self.assertEqual(model_basic.name, "model")
        self.assertEqual(model_basic_1.name, "model_1")
        self.assertEqual(model_basic_2.name, "model_2")
        self.assertEqual(model_basic_3.name, "model_3")
        self.assertEqual(model_basic_given_name.name, "a_static_model")

        try:
            model_basic_given_repeat_name = basic_static_model("model_1")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect repeat user given names")

        model_nested = nested_static_model()
        model_nested_1 = nested_static_model(inner_model_name="a_inner_static_model")

        self.assertEqual(model_nested.name, "model_5")
        self.assertEqual(model_nested_1.name, "model_6")

        try:
            model_nested_given_repeat_name = nested_static_model(inner_model_name="a_inner_static_model")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect repeat user given names")

    def test_layer_name_uniqueness(self):
        print('-' * 20, 'test_layer_name_uniqueness', '-' * 20)
        test_flag =True

        # dynamic
        try:
            model_dynamic = basic_dynamic_model(conv1_name="conv", conv2_name="conv")
            # dynamic mode check uniqueness when self.all_layers is called
            all_layers = model_dynamic.all_layers
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect that layers inside a model have the same name in dynamic mode")

        # static
        try:
            model_static = basic_static_model(conv1_name="conv", conv2_name="conv")
            print(model_static)
            print("$$$$$$$$$$$$$$$$$$")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect that layers inside a model have the same name in static mode")

    def test_vgg_auto_naming(self):
        print('-' * 20, 'test_vgg_auto_naming', '-' * 20)
        test_flag = True

        vgg = VGG16()
        vgg_1 = VGG16()
        vgg_2 = VGG16(name="vgg16_2")
        vgg_3 = VGG16()
        vgg_given_name = VGG16(name="a_vgg_model")

        self.assertEqual(vgg.name, "vgg16")
        self.assertEqual(vgg_1.name, "vgg16_1")
        self.assertEqual(vgg_2.name, "vgg16_2")
        self.assertEqual(vgg_3.name, "vgg16_3")
        self.assertEqual(vgg_given_name.name, "a_vgg_model")

        try:
            vgg_given_repeat_name = VGG16(name="vgg16_1")
            test_flag = False
        except Exception as e:
            pass
        if not test_flag:
            self.fail("Failed to detect repeat user given names")

    def test_layerlist(self):
        test_flag = True

        try:
            inputs = tl.layers.Input([10, 5])
            layer1 = tl.layers.LayerList([
                tl.layers.Dense(n_units=4, name='dense1'),
                tl.layers.Dense(n_units=3, name='dense1')
            ])(inputs)
            model = tl.models.Model(inputs=inputs, outputs=layer1, name='layerlistmodel')
            print(model)
            print([w.name for w in model.weights])
            test_flag = False
        except Exception as e:
            print(e)
        if not test_flag:
            self.fail("Fail to detect duplicate name in layerlist")


if __name__ == '__main__':
    unittest.main()
