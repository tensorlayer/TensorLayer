#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Merge_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.data = dict()

        ##############
        #   vector   #
        ##############

        x = tf.placeholder(tf.float32, shape=[None, 784])
        inputs = tl.layers.InputLayer(x, name='input_layer')

        net_v1_1 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='relu1_1')
        net_v1_2 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='relu2_1')
        net_v1 = tl.layers.ConcatLayer([net_v1_1, net_v1_2], concat_dim=1, name='concat_layer')

        net_v1.print_params(False)
        net_v1.print_layers()

        cls.data["net_vector1"] = dict()
        cls.data["net_vector1"]["layers"] = net_v1.all_layers
        cls.data["net_vector1"]["params"] = net_v1.all_params
        cls.data["net_vector1"]["n_params"] = net_v1.count_params()

        net_v2_1 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_0')
        net_v2_2 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_1')
        net_v2 = tl.layers.ElementwiseLayer([net_v2_1, net_v2_2], combine_fn=tf.minimum, name='minimum')

        net_v2.print_params(False)
        net_v2.print_layers()

        cls.data["net_vector2"] = dict()
        cls.data["net_vector2"]["layers"] = net_v2.all_layers
        cls.data["net_vector2"]["params"] = net_v2.all_params
        cls.data["net_vector2"]["n_params"] = net_v2.count_params()

        net_v3_1 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_a')
        net_v3_2 = tl.layers.DenseLayer(inputs, n_units=100, act=tf.nn.relu, name='net_b')
        net_v3 = tl.layers.ElementwiseLambdaLayer([net_v3_1, net_v3_2], fn=lambda a, b: a * b, name='multiply')

        net_v3.print_params(False)
        net_v3.print_layers()

        cls.data["net_vector3"] = dict()
        cls.data["net_vector3"]["layers"] = net_v3.all_layers
        cls.data["net_vector3"]["params"] = net_v3.all_params
        cls.data["net_vector3"]["n_params"] = net_v3.count_params()

        #############
        #   Image   #
        #############

        x = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
        inputs = tl.layers.InputLayer(x, name='input')

        net_im1_1 = tl.layers.Conv2d(inputs, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='c1')
        net_im1_2 = tl.layers.Conv2d(inputs, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='c2')
        net_im1 = tl.layers.ConcatLayer([net_im1_1, net_im1_2], concat_dim=-1, name='concat')

        net_im1.print_params(False)
        net_im1.print_layers()

        cls.data["net_image1"] = dict()
        cls.data["net_image1"]["shape"] = net_im1.outputs.get_shape().as_list()
        cls.data["net_image1"]["layers"] = net_im1.all_layers
        cls.data["net_image1"]["params"] = net_im1.all_params
        cls.data["net_image1"]["n_params"] = net_im1.count_params()

        net_im2 = tl.layers.ElementwiseLayer([net_im1_1, net_im1_2], combine_fn=tf.minimum, name='minimum2')

        net_im2.print_params(False)
        net_im2.print_layers()

        cls.data["net_image2"] = dict()
        cls.data["net_image2"]["shape"] = net_im2.outputs.get_shape().as_list()
        cls.data["net_image2"]["layers"] = net_im2.all_layers
        cls.data["net_image2"]["params"] = net_im2.all_params
        cls.data["net_image2"]["n_params"] = net_im2.count_params()

        net_im3 = tl.layers.ElementwiseLambdaLayer([net_im1_1, net_im1_2], fn=lambda a, b: a * b, name='multiply2')

        net_im3.print_params(False)
        net_im3.print_layers()

        cls.data["net_image3"] = dict()
        cls.data["net_image3"]["shape"] = net_im3.outputs.get_shape().as_list()
        cls.data["net_image3"]["layers"] = net_im3.all_layers
        cls.data["net_image3"]["params"] = net_im3.all_params
        cls.data["net_image3"]["n_params"] = net_im3.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net_vector1(self):
        self.assertEqual(len(self.data["net_vector1"]["layers"]), 4)
        self.assertEqual(len(self.data["net_vector1"]["params"]), 4)
        self.assertEqual(self.data["net_vector1"]["n_params"], 157000)

    def test_net_vector2(self):
        self.assertEqual(len(self.data["net_vector2"]["layers"]), 4)
        self.assertEqual(len(self.data["net_vector2"]["params"]), 4)
        self.assertEqual(self.data["net_vector2"]["n_params"], 157000)

    def test_net_vector3(self):
        self.assertEqual(len(self.data["net_vector3"]["layers"]), 4)
        self.assertEqual(len(self.data["net_vector3"]["params"]), 4)
        self.assertEqual(self.data["net_vector3"]["n_params"], 157000)

    def test_net_image1(self):
        self.assertEqual(self.data["net_image1"]["shape"][1:], [50, 50, 64])
        self.assertEqual(len(self.data["net_image1"]["layers"]), 4)
        self.assertEqual(len(self.data["net_image1"]["params"]), 4)
        self.assertEqual(self.data["net_image1"]["n_params"], 1792)

    def test_net_image2(self):
        self.assertEqual(self.data["net_image2"]["shape"][1:], [50, 50, 32])
        self.assertEqual(len(self.data["net_image2"]["layers"]), 4)
        self.assertEqual(len(self.data["net_image2"]["params"]), 4)
        self.assertEqual(self.data["net_image2"]["n_params"], 1792)

    def test_net_image3(self):
        self.assertEqual(self.data["net_image3"]["shape"][1:], [50, 50, 32])
        self.assertEqual(len(self.data["net_image3"]["layers"]), 4)
        self.assertEqual(len(self.data["net_image3"]["params"]), 4)
        self.assertEqual(self.data["net_image3"]["n_params"], 1792)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
