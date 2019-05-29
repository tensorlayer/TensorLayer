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


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("##### begin testing activation #####")

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_layer_core_act(cls):

        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 400, 400, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.n1 = tl.layers.Conv2dLayer(
            act=tf.nn.relu, shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
            b_init=tf.constant_initializer(value=0.0), name='conv2dlayer'
        )(cls.input_layer)

        cls.n2 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="relu", name='conv2d')(cls.n1)

        cls.n3 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="leaky_relu",
                                  b_init=None)(cls.n2)

        cls.n4 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="lrelu", b_init=None)(cls.n2)

        cls.n5 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="sigmoid",
                                  in_channels=32)(cls.n4)

        cls.n6 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="tanh", in_channels=32)(cls.n5)

        cls.n7 = tl.layers.Conv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), act="leaky_relu0.22", in_channels=32
        )(cls.n6)

        cls.n8 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="lrelu0.22",
                                  in_channels=32)(cls.n7)

        cls.n9 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="softplus",
                                  in_channels=32)(cls.n8)

        cls.n10 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act="relu6", in_channels=32)(cls.n9)

        cls.model = Model(cls.input_layer, cls.n8)


class Exception_test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("##### begin testing exception in activation #####")

    def test_exception(cls):

        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 400, 400, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        try:
            cls.n1 = tl.layers.Conv2dLayer(
                act='activation', shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
                b_init=tf.constant_initializer(value=0.0), name='conv2dlayer'
            )(cls.input_layer)
        except Exception as e:
            cls.assertIsInstance(e, Exception)
            print(e)

        try:
            cls.n2 = tl.layers.Conv2dLayer(
                act='leaky_relu0.2x', shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
                b_init=tf.constant_initializer(value=0.0), name='conv2dlayer'
            )(cls.input_layer)
        except Exception as e:
            cls.assertIsInstance(e, Exception)
            print(e)

        try:
            cls.n3 = tl.layers.Conv2dLayer(
                act='lrelu0.2x', shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
                b_init=tf.constant_initializer(value=0.0), name='conv2dlayer'
            )(cls.input_layer)
        except Exception as e:
            cls.assertIsInstance(e, Exception)
            print(e)


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
