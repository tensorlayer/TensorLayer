#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

import tensorflow as tf
import tensorlayer as tl


class Layer_Convolution_Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ############
        #    1D    #
        ############

        x1 = tf.placeholder(tf.float32, (None, 100, 1))
        nin1 = tl.layers.InputLayer(x1, name='in1')

        n1 = tl.layers.Conv1dLayer(nin1, shape=(5, 1, 32), stride=2)
        cls.shape_n1 = n1.outputs.get_shape().as_list()

        n2 = tl.layers.Conv1d(nin1, n_filter=32, filter_size=5, stride=2)
        cls.shape_n2 = n2.outputs.get_shape().as_list()

        n2_1 = tl.layers.SeparableConv1d(
            nin1, n_filter=32, filter_size=3, strides=1, padding='VALID', act=tf.nn.relu, name='seperable1d1'
        )
        cls.shape_n2_1 = n2_1.outputs.get_shape().as_list()
        cls.n2_1_all_layers = n2_1.all_layers
        cls.n2_1_params = n2_1.all_params
        cls.n2_1_count_params = n2_1.count_params()

        ############
        #    2D    #
        ############

        x2 = tf.placeholder(tf.float32, (None, 100, 100, 3))
        nin2 = tl.layers.InputLayer(x2, name='in2')

        n3 = tl.layers.Conv2dLayer(
            nin2, act=tf.nn.relu, shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=5e-2), b_init=tf.constant_initializer(value=0.0),
            name='conv2dlayer'
        )
        cls.shape_n3 = n3.outputs.get_shape().as_list()

        n4 = tl.layers.Conv2d(nin2, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d')
        cls.shape_n4 = n4.outputs.get_shape().as_list()
        cls.n4_params = n4.all_params

        n5 = tl.layers.Conv2d(
            nin2, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, b_init=None, name='conv2d_no_bias'
        )
        cls.shape_n5 = n5.outputs.get_shape().as_list()
        cls.n5_params = n5.all_params

        n6 = tl.layers.DeConv2dLayer(
            nin2, shape=(5, 5, 32, 3), output_shape=(100, 200, 200, 32), strides=(1, 2, 2, 1), name='deconv2dlayer'
        )
        cls.shape_n6 = n6.outputs.get_shape().as_list()

        n7 = tl.layers.DeConv2d(nin2, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d')
        cls.shape_n7 = n7.outputs.get_shape().as_list()

        n8 = tl.layers.DepthwiseConv2d(
            nin2, shape=(3, 3), strides=(2, 2), act=tf.nn.relu, depth_multiplier=2, name='depthwise'
        )
        cls.shape_n8 = n8.outputs.get_shape().as_list()

        n9 = tl.layers.Conv2d(nin2, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='conv2d2')
        n9 = tl.layers.GroupConv2d(n9, n_filter=32, filter_size=(3, 3), strides=(2, 2), name='group')
        cls.shape_n9 = n9.outputs.get_shape().as_list()

        n10 = tl.layers.SeparableConv2d(
            nin2, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, name='seperable2d1'
        )
        cls.shape_n10 = n10.outputs.get_shape().as_list()
        cls.n10_all_layers = n10.all_layers
        cls.n10_params = n10.all_params
        cls.n10_count_params = n10.count_params()

        ############
        #    3D    #
        ############

        x3 = tf.placeholder(tf.float32, (None, 100, 100, 100, 3))
        nin3 = tl.layers.InputLayer(x3, name='in3')

        n11 = tl.layers.Conv3dLayer(nin3, shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))
        cls.shape_n11 = n11.outputs.get_shape().as_list()

        # n = tl.layers.DeConv3dLayer(nin, shape=(2, 2, 2, 128, 3), output_shape=(100, 12, 32, 32, 128), strides=(1, 2, 2, 2, 1))
        # print(n)
        # shape = n.outputs.get_shape().as_list()

        n12 = tl.layers.DeConv3d(nin3, n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))
        cls.shape_n12 = n12.outputs.get_shape().as_list()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_shape_n1(self):
        self.assertEqual(self.shape_n1[1], 50)
        self.assertEqual(self.shape_n1[2], 32)

    def test_shape_n2(self):
        self.assertEqual(self.shape_n2[1], 50)
        self.assertEqual(self.shape_n2[2], 32)

    def test_shape_n2_1(self):
        self.assertEqual(self.shape_n2_1[1], 98)
        self.assertEqual(self.shape_n2_1[2], 32)

    def test_shape_n3(self):
        self.assertEqual(self.shape_n3[1], 50)
        self.assertEqual(self.shape_n3[2], 50)
        self.assertEqual(self.shape_n3[3], 32)

    def test_shape_n4(self):
        self.assertEqual(self.shape_n4[1], 50)
        self.assertEqual(self.shape_n4[2], 50)
        self.assertEqual(self.shape_n4[3], 32)

    def test_shape_n5(self):
        self.assertEqual(self.shape_n5[1], 50)
        self.assertEqual(self.shape_n5[2], 50)
        self.assertEqual(self.shape_n5[3], 32)

    def test_shape_n6(self):
        self.assertEqual(self.shape_n6[1], 200)
        self.assertEqual(self.shape_n6[2], 200)
        self.assertEqual(self.shape_n6[3], 32)

    def test_shape_n7(self):
        #self.assertEqual(self.shape_n7[1], 200)  # TODO: why [None None None 32] ?
        #self.assertEqual(self.shape_n7[2], 200)  # TODO: why [None None None 32] ?
        self.assertEqual(self.shape_n7[3], 32)

    def test_shape_n8(self):
        self.assertEqual(self.shape_n8[1], 50)
        self.assertEqual(self.shape_n8[2], 50)
        self.assertEqual(self.shape_n8[3], 6)

    def test_shape_n9(self):
        self.assertEqual(self.shape_n9[1], 25)
        self.assertEqual(self.shape_n9[2], 25)
        self.assertEqual(self.shape_n9[3], 32)

    def test_shape_n10(self):
        self.assertEqual(self.shape_n10[1:], [98, 98, 32])

    def test_shape_n11(self):
        self.assertEqual(self.shape_n11[1], 50)
        self.assertEqual(self.shape_n11[2], 50)
        self.assertEqual(self.shape_n11[3], 50)
        self.assertEqual(self.shape_n11[4], 32)

    def test_shape_n12(self):
        self.assertEqual(self.shape_n12[1], 200)
        self.assertEqual(self.shape_n12[2], 200)
        self.assertEqual(self.shape_n12[3], 200)
        self.assertEqual(self.shape_n12[4], 32)

    def test_params_n2_1(self):
        self.assertEqual(len(self.n2_1_params), 3)

    def test_params_n4(self):
        self.assertEqual(len(self.n4_params), 2)

    def test_params_n5(self):
        self.assertEqual(len(self.n5_params), 1)

    def test_params_n10(self):
        self.assertEqual(len(self.n10_params), 3)
        self.assertEqual(self.n10_count_params, 155)

    def test_layers_n2_1(self):
        self.assertEqual(len(self.n2_1_all_layers), 1)

    def test_layers_n10(self):
        self.assertEqual(len(self.n10_all_layers), 1)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()
