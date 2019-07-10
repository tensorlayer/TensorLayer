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
    M = Model(inputs=ni, outputs=nn)
    return M


class basic_dynamic_model(Model):

    def __init__(self):
        super(basic_dynamic_model, self).__init__()
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
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dynamic_basic(self):
        print('-' * 20, 'test_dynamic_basic', '-' * 20)
        model_basic = basic_dynamic_model()

        # test empty model before calling
        self.assertEqual(model_basic.is_train, None)
        self.assertEqual(model_basic._all_weights, None)
        self.assertEqual(model_basic._inputs, None)
        self.assertEqual(model_basic._outputs, None)
        self.assertEqual(model_basic._model_layer, None)
        self.assertEqual(model_basic._all_layers, None)
        self.assertEqual(model_basic._nodes_fixed, False)

        # test layer and weights access
        all_layers = model_basic.all_layers
        self.assertEqual(len(model_basic.all_layers), 7)
        self.assertEqual(model_basic._all_weights, None)

        self.assertIsNotNone(model_basic.all_weights)
        print([w.name for w in model_basic.all_weights])

        # test model mode
        model_basic.train()
        self.assertEqual(model_basic.is_train, True)
        model_basic.eval()
        self.assertEqual(model_basic.is_train, False)
        model_basic.test()
        self.assertEqual(model_basic.is_train, False)
        model_basic.infer()
        self.assertEqual(model_basic.is_train, False)

        # test as_layer
        try:
            model_basic.as_layer()
        except Exception as e:
            print(e)
        self.assertIsNone(model_basic._model_layer)

        # test print
        try:
            print(model_basic)
        except Exception as e:
            print(e)

        # test forwarding
        inputs = np.random.normal(size=[2, 24, 24, 3]).astype(np.float32)
        outputs1 = model_basic(inputs)
        self.assertEqual(model_basic._nodes_fixed, True)
        self.assertEqual(model_basic.is_train, False)

        try:
            outputs2 = model_basic(inputs, is_train=True)
        except Exception as e:
            print(e)
        outputs2 = model_basic(inputs, is_train=False)
        self.assertEqual(model_basic.is_train, False)

        self.assertLess(np.max(np.abs(outputs1.numpy() - outputs2.numpy())), 1e-7)

        # test layer node
        self.assertEqual(len(model_basic.all_layers[-1]._nodes), 0)
        self.assertEqual(model_basic.all_layers[-2]._nodes_fixed, True)

        # test release_memory
        try:
            model_basic.release_memory()
        except Exception as e:
            print(e)

    def test_static_basic(self):
        print('-' * 20, 'test_static_basic', '-' * 20)
        model_basic = basic_static_model()

        # test empty model before calling
        self.assertEqual(model_basic.is_train, None)
        self.assertEqual(model_basic._all_weights, None)
        self.assertIsNotNone(model_basic._inputs)
        self.assertIsNotNone(model_basic._outputs)
        self.assertEqual(model_basic._model_layer, None)
        self.assertIsNotNone(model_basic._all_layers)
        self.assertIsNotNone(model_basic._nodes_fixed)

        # test layer and weights access
        all_layers = model_basic.all_layers
        self.assertEqual(len(model_basic.all_layers), 8)
        self.assertEqual(model_basic._all_weights, None)

        self.assertIsNotNone(model_basic.all_weights)
        print([w.name for w in model_basic.all_weights])

        # test model mode
        model_basic.train()
        self.assertEqual(model_basic.is_train, True)
        model_basic.eval()
        self.assertEqual(model_basic.is_train, False)
        model_basic.test()
        self.assertEqual(model_basic.is_train, False)
        model_basic.infer()
        self.assertEqual(model_basic.is_train, False)

        # test as_layer
        self.assertIsInstance(model_basic.as_layer(), tl.layers.Layer)
        self.assertIsNotNone(model_basic._model_layer)

        # test print
        try:
            print(model_basic)
        except Exception as e:
            print(e)

        # test forwarding
        inputs = np.random.normal(size=[2, 24, 24, 3]).astype(np.float32)
        outputs1 = model_basic(inputs)
        self.assertEqual(model_basic._nodes_fixed, True)
        self.assertEqual(model_basic.is_train, False)

        try:
            outputs2 = model_basic(inputs, is_train=True)
        except Exception as e:
            print(e)
        outputs2 = model_basic(inputs, is_train=False)
        self.assertEqual(model_basic.is_train, False)

        self.assertLess(np.max(np.abs(outputs1.numpy() - outputs2.numpy())), 1e-7)

        # test layer node
        self.assertEqual(len(model_basic.all_layers[-1]._nodes), 1)
        self.assertEqual(model_basic.all_layers[-2]._nodes_fixed, True)

        # test release_memory
        try:
            model_basic.release_memory()
        except Exception as e:
            print(e)

    def test_deprecated_function(self):
        print('-' * 20, 'test_deprecated_function', '-' * 20)
        model = basic_dynamic_model()

        try:
            model.print_all_layers()
        except Exception as e:
            print(e)

        try:
            model.count_params()
        except Exception as e:
            print(e)

        try:
            model.print_params()
        except Exception as e:
            print(e)

        try:
            model.all_params()
        except Exception as e:
            print(e)

        try:
            model.all_drop()
        except Exception as e:
            print(e)

    def test_exceptions(self):
        print('-' * 20, 'test exceptions', '-' * 20)
        np_arr = np.random.normal(size=[4, 784]).astype(np.float32)
        tf_tensor = tf.random.normal(shape=[4, 784])
        ni = Input(shape=[4, 784])

        try:
            model = Model(inputs=[], outputs=[])
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

        try:
            model = Model(inputs=np_arr, outputs=np_arr + 1)
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            print(e)

        try:
            model = Model(inputs=[np_arr], outputs=[np_arr + 1])
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            print(e)

        try:
            model = Model(inputs=[tf_tensor], outputs=[tf_tensor + 1])
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            print(e)

        try:
            model = Model(inputs=tf_tensor, outputs=[tf_tensor + 1])
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            print(e)

        try:
            model = Model(inputs=ni, outputs=[tf_tensor + 1])
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            print(e)

        try:

            class ill_model(Model):

                def __init__(self):
                    super(ill_model, self).__init__()
                    self.dense2 = Dense(10, act=None)

                def forward(self, x):
                    x = self.dense2(x)
                    return x

            model = ill_model()
            weights = model.all_weights
        except Exception as e:
            self.assertIsInstance(e, AttributeError)
            print(e)

        try:
            ni = Input([4, 784])
            nn = Dense(10)(ni)
            model = Model(inputs=ni, outputs=nn)
            outputs = model(np_arr)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

        try:
            ni = Input([4, 784])
            model = Model(inputs=ni, outputs=ni)
            model.save_weights('./empty_model.h5')
        except Exception as e:
            print(e)

        try:
            ni = Input([4, 784])
            nn = Dense(10)(ni)
            model = Model(inputs=ni, outputs=nn)
            model._outputs = None
            outputs = model(np_arr, is_train=True)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

    def test_list_inputs_outputs(self):
        print('-' * 20, 'test_list_inputs_outputs', '-' * 20)
        ni_1 = Input(shape=[4, 16])
        ni_2 = Input(shape=[4, 32])
        a_1 = Dense(80)(ni_1)
        b_1 = Dense(160)(ni_2)
        concat = Concat()([a_1, b_1])
        a_2 = Dense(10)(concat)
        b_2 = Dense(20)(concat)

        model = Model(inputs=[ni_1, ni_2], outputs=[a_2, b_2])

        model.train()
        np_arr1 = np.random.normal(size=[4, 16]).astype(np.float32)
        np_arr2 = np.random.normal(size=[4, 32]).astype(np.float32)

        try:
            outputs = model(np_arr1)
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

        try:
            outputs = model([np_arr1])
        except Exception as e:
            self.assertIsInstance(e, ValueError)
            print(e)

        out_a, out_b = model([np_arr1, np_arr2])
        self.assertEqual(out_a.shape, [4, 10])
        self.assertEqual(out_b.shape, [4, 20])

    def test_special_case(self):
        print('-' * 20, 'test_special_case', '-' * 20)

        class my_model(Model):

            def __init__(self):
                super(my_model, self).__init__()
                self.dense = Dense(64, in_channels=3)
                self.vgg = tl.models.vgg16()

            def forward(self, x):
                return x

        model = my_model()
        weights = model.all_weights
        self.assertGreater(len(weights), 2)
        print(len(weights))

    def test_get_layer(self):
        print('-' * 20, 'test_get_layer', '-' * 20)
        model_basic = basic_dynamic_model()
        self.assertIsInstance(model_basic.get_layer('conv2'), tl.layers.Conv2d)
        try:
            model_basic.get_layer('abc')
        except Exception as e:
            print(e)

        try:
            model_basic.get_layer(index=99)
        except Exception as e:
            print(e)

        model_basic = basic_static_model()
        self.assertIsInstance(model_basic.get_layer('conv2'), tl.layers.Conv2d)
        self.assertIsInstance(model_basic.get_layer(index=2), tl.layers.MaxPool2d)
        print([w.name for w in model_basic.get_layer(index=-1).all_weights])
        try:
            model_basic.get_layer('abc')
        except Exception as e:
            print(e)

        try:
            model_basic.get_layer(index=99)
        except Exception as e:
            print(e)

    def test_model_weights_copy(self):
        print('-' * 20, 'test_model_weights_copy', '-' * 20)
        model_basic = basic_static_model()
        model_weights = model_basic.trainable_weights
        ori_len = len(model_weights)
        model_weights.append(np.arange(5))
        new_len = len(model_weights)
        self.assertEqual(new_len - 1, ori_len)

    def test_inchannels_exception(self):
        print('-' * 20, 'test_inchannels_exception', '-' * 20)

        class my_model(Model):

            def __init__(self):
                super(my_model, self).__init__()
                self.dense = Dense(64)
                self.vgg = tl.models.vgg16()

            def forward(self, x):
                return x

        try:
            M = my_model()
        except Exception as e:
            self.assertIsInstance(e, AttributeError)
            print(e)


if __name__ == '__main__':

    unittest.main()
