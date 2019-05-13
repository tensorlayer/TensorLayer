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


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 8
        cls.inputs_shape = [cls.batch_size, 100, 1]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.n1 = tl.layers.Conv1dLayer(shape=(5, 1, 32), stride=2)(cls.input_layer)

        cls.n2 = tl.layers.Conv1d(n_filter=32, filter_size=5, stride=2)(cls.n1)

        cls.n3 = tl.layers.DeConv1dLayer(
            shape=(5, 64, 32), outputs_shape=(cls.batch_size, 50, 64), strides=(1, 2, 1), name='deconv1dlayer'
        )(cls.n2)

        cls.n4 = tl.layers.SeparableConv1d(
            n_filter=32, filter_size=3, strides=2, padding='SAME', act=tf.nn.relu, name='separable_1d'
        )(cls.n3)

        cls.n5 = tl.layers.SubpixelConv1d(scale=2, act=tf.nn.relu, in_channels=32, name='subpixel_1d')(cls.n4)

        cls.model = Model(inputs=cls.input_layer, outputs=cls.n5)
        print("Testing Conv1d model: \n", cls.model)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_layer_n1(self):

        # self.assertEqual(len(self.n1.all_layers), 2)
        # self.assertEqual(len(self.n1.all_params), 2)
        # self.assertEqual(self.n1.count_params(), 192)
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [50, 32])

    def test_layer_n2(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 5344)
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [25, 32])

    def test_layer_n3(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 5344)
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 2)
        self.assertEqual(self.n3.get_shape().as_list()[1:], [50, 64])

    def test_layer_n4(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 5344)
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 3)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [25, 32])

    def test_layer_n5(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 5344)
        self.assertEqual(self.n5.get_shape().as_list()[1:], [50, 16])

    # def test_layer_n3(self):
    #
    #     self.assertEqual(len(self.n3.all_layers), 4)
    #     self.assertEqual(len(self.n3.all_params), 7)
    #     self.assertEqual(self.n3.count_params(), 6496)
    #     self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [23, 32])


# FIXME: TF2.0 only supports NHWC now
# class Layer_Convolution_1D_NCW_Test(CustomTestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         print("\n#################################")
#
#         cls.batch_size = 8
#         cls.inputs_shape = [cls.batch_size, 1, 100]
#         cls.input_layer = Input(cls.inputs_shape, name='input_layer')
#
#         cls.n1 = tl.layers.Conv1dLayer(
#             shape=(5, 1, 32), stride=2, data_format="NCW"
#         )(cls.input_layer)
#         cls.n2 = tl.layers.Conv1d(
#             n_filter=32, filter_size=5, stride=2, data_format='channels_first'
#         )(cls.n1)
#         cls.model = Model(inputs=cls.input_layer, outputs=cls.n2)
#         print("Testing Conv1d model: \n", cls.model)
#
#         # cls.n3 = tl.layers.SeparableConv1d(
#         #     cls.n2, n_filter=32, filter_size=3, strides=1, padding='VALID', act=tf.nn.relu, name='separable_1d'
#         # )
#
#     @classmethod
#     def tearDownClass(cls):
#         pass
#         # tf.reset_default_graph()
#
#     def test_layer_n1(self):
#
#         # self.assertEqual(len(self.n1.all_layers), 2)
#         # self.assertEqual(len(self.n1.all_params), 2)
#         # self.assertEqual(self.n1.count_params(), 192)
#         self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
#         self.assertEqual(self.n1.get_shape().as_list()[1:], [50, 32])
#
#     def test_layer_n2(self):
#
#         # self.assertEqual(len(self.n2.all_layers), 3)
#         # self.assertEqual(len(self.n2.all_params), 4)
#         # self.assertEqual(self.n2.count_params(), 5344)
#         self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
#         self.assertEqual(self.n2.get_shape().as_list()[1:], [25, 32])
#
#     # def test_layer_n3(self):
#     #
#     #     self.assertEqual(len(self.n3.all_layers), 4)
#     #     self.assertEqual(len(self.n3.all_params), 7)
#     #     self.assertEqual(self.n3.count_params(), 6496)
#     #     self.assertEqual(self.n3.outputs.get_shape().as_list()[1:], [23, 32])


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 400, 400, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.n1 = tl.layers.Conv2dLayer(
            act=tf.nn.relu, shape=(5, 5, 3, 32), strides=(1, 2, 2, 1), padding='SAME',
            b_init=tf.constant_initializer(value=0.0), name='conv2dlayer'
        )(cls.input_layer)

        cls.n2 = tl.layers.Conv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d')(cls.n1)

        cls.n3 = tl.layers.Conv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, b_init=None, name='conv2d_no_bias'
        )(cls.n2)

        cls.n4 = tl.layers.DeConv2dLayer(
            shape=(5, 5, 32, 32), outputs_shape=(cls.batch_size, 100, 100, 32), strides=(1, 2, 2, 1),
            name='deconv2dlayer'
        )(cls.n3)

        cls.n5 = tl.layers.DeConv2d(n_filter=32, filter_size=(3, 3), strides=(2, 2), name='DeConv2d')(cls.n4)

        cls.n6 = tl.layers.DepthwiseConv2d(
            filter_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), act=tf.nn.relu, depth_multiplier=2,
            name='depthwise'
        )(cls.n5)

        cls.n7 = tl.layers.Conv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, in_channels=64, name='conv2d2'
        )(cls.n6)

        cls.n8 = tl.layers.BinaryConv2d(
            n_filter=64, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, in_channels=32, name='binaryconv2d'
        )(cls.n7)

        cls.n9 = tl.layers.SeparableConv2d(
            n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tf.nn.relu, name='separableconv2d'
        )(cls.n8)

        cls.n10 = tl.layers.GroupConv2d(n_filter=64, filter_size=(3, 3), strides=(2, 2), n_group=2,
                                        name='group')(cls.n9)

        cls.n11 = tl.layers.DorefaConv2d(
            n_filter=32, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='dorefaconv2d'
        )(cls.n10)

        cls.n12 = tl.layers.TernaryConv2d(
            n_filter=64, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='ternaryconv2d'
        )(cls.n11)

        cls.n13 = tl.layers.QuanConv2d(
            n_filter=32, filter_size=(5, 5), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='quancnn2d'
        )(cls.n12)

        cls.n14 = tl.layers.SubpixelConv2d(scale=2, act=tf.nn.relu, name='subpixelconv2d')(cls.n13)

        cls.model = Model(cls.input_layer, cls.n14)
        print("Testing Conv2d model: \n", cls.model)

        # cls.n12 = tl.layers.QuanConv2d(cls.n11, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='quancnn')

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_layer_n1(self):

        # self.assertEqual(len(self.n1.all_layers), 2)
        # self.assertEqual(len(self.n1.all_params), 2)
        # self.assertEqual(self.n1.count_params(), 2432)
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [200, 200, 32])

    def test_layer_n2(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 11680)
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n3(self):

        # self.assertEqual(len(self.n3.all_layers), 4)
        # self.assertEqual(len(self.n3.all_params), 5)
        # self.assertEqual(self.n3.count_params(), 20896)
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 1)  # b_init is None
        self.assertEqual(self.n3.get_shape().as_list()[1:], [50, 50, 32])

    def test_layer_n4(self):

        # self.assertEqual(len(self.n4.all_layers), 5)
        # self.assertEqual(len(self.n4.all_params), 7)
        # self.assertEqual(self.n4.count_params(), 46528)
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 2)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n5(self):

        # self.assertEqual(len(self.n5.all_layers), 6)
        # self.assertEqual(len(self.n5.all_params), 9)
        # self.assertEqual(self.n5.count_params(), 55776)
        self.assertEqual(len(self.n5._info[0].layer.all_weights), 2)
        self.assertEqual(self.n5.get_shape().as_list()[1:], [200, 200, 32])

    def test_layer_n6(self):

        # self.assertEqual(len(self.n6.all_layers), 7)
        # self.assertEqual(len(self.n6.all_params), 11)
        # self.assertEqual(self.n6.count_params(), 56416)
        self.assertEqual(len(self.n6._info[0].layer.all_weights), 2)
        self.assertEqual(self.n6.get_shape().as_list()[1:], [200, 200, 64])

    def test_layer_n7(self):

        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n7._info[0].layer.all_weights), 2)
        self.assertEqual(self.n7.get_shape().as_list()[1:], [100, 100, 32])

    def test_layer_n8(self):

        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n8._info[0].layer.all_weights), 2)
        self.assertEqual(self.n8.get_shape().as_list()[1:], [50, 50, 64])

    def test_layer_n9(self):

        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n9._info[0].layer.all_weights), 3)
        self.assertEqual(self.n9.get_shape().as_list()[1:], [24, 24, 32])

    def test_layer_n10(self):
        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n10._info[0].layer.all_weights), 2)
        self.assertEqual(self.n10.get_shape().as_list()[1:], [12, 12, 64])

    def test_layer_n11(self):
        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n11._info[0].layer.all_weights), 2)
        self.assertEqual(self.n11.get_shape().as_list()[1:], [12, 12, 32])

    def test_layer_n12(self):
        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n12._info[0].layer.all_weights), 2)
        self.assertEqual(self.n12.get_shape().as_list()[1:], [12, 12, 64])

    def test_layer_n13(self):
        # self.assertEqual(len(self.n7.all_layers), 8)
        # self.assertEqual(len(self.n7.all_params), 13)
        # self.assertEqual(self.n7.count_params(), 74880)
        self.assertEqual(len(self.n13._info[0].layer.all_weights), 2)
        self.assertEqual(self.n13.get_shape().as_list()[1:], [12, 12, 32])

    def test_layer_n14(self):
        self.assertEqual(self.n14.get_shape().as_list()[1:], [24, 24, 8])

    # def test_layer_n8(self):
    #
    #     self.assertEqual(len(self.n8.all_layers), 9)
    #     self.assertEqual(len(self.n8.all_params), 15)
    #     self.assertEqual(self.n8.count_params(), 79520)
    #     self.assertEqual(self.n8.outputs.get_shape().as_list()[1:], [50, 50, 32])
    #
    # def test_layer_n9(self):
    #
    #     self.assertEqual(len(self.n9.all_layers), 10)
    #     self.assertEqual(len(self.n9.all_params), 18)
    #     self.assertEqual(self.n9.count_params(), 80864)
    #     self.assertEqual(self.n9.outputs.get_shape().as_list()[1:], [48, 48, 32])
    #
    # def test_layer_n10(self):
    #
    #     self.assertEqual(len(self.n10.all_layers), 11)
    #     self.assertEqual(len(self.n10.all_params), 20)
    #     self.assertEqual(self.n10.count_params(), 132128)
    #     self.assertEqual(self.n10.outputs.get_shape().as_list()[1:], [48, 48, 64])
    #
    # def test_layer_n11(self):
    #
    #     self.assertEqual(len(self.n11.all_layers), 12)
    #     self.assertEqual(len(self.n11.all_params), 22)
    #     self.assertEqual(self.n11.count_params(), 150592)
    #     self.assertEqual(self.n11.outputs.get_shape().as_list()[1:], [96, 96, 32])
    #
    # def test_layer_n12(self):
    #
    #     self.assertEqual(len(self.n12.all_layers), 13)
    #     self.assertEqual(len(self.n12.all_params), 24)
    #     self.assertEqual(self.n12.count_params(), 201856)
    #     self.assertEqual(self.n12.outputs.get_shape().as_list()[1:], [96, 96, 64])


class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        print("\n#################################")

        cls.batch_size = 5
        cls.inputs_shape = [cls.batch_size, 20, 20, 20, 3]
        cls.input_layer = Input(cls.inputs_shape, name='input_layer')

        cls.n1 = tl.layers.Conv3dLayer(shape=(2, 2, 2, 3, 32), strides=(1, 2, 2, 2, 1))(cls.input_layer)

        cls.n2 = tl.layers.DeConv3dLayer(
            shape=(2, 2, 2, 128, 32), outputs_shape=(cls.batch_size, 20, 20, 20, 128), strides=(1, 2, 2, 2, 1)
        )(cls.n1)

        cls.n3 = tl.layers.Conv3d(
            n_filter=64, filter_size=(3, 3, 3), strides=(3, 3, 3), act=tf.nn.relu, b_init=None, in_channels=128,
            name='conv3d_no_bias'
        )(cls.n2)

        cls.n4 = tl.layers.DeConv3d(n_filter=32, filter_size=(3, 3, 3), strides=(2, 2, 2))(cls.n3)

        cls.model = Model(inputs=cls.input_layer, outputs=cls.n4)
        print("Testing Conv3d model: \n", cls.model)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_layer_n1(self):

        # self.assertEqual(len(self.n1.all_layers), 2)
        # self.assertEqual(len(self.n1.all_params), 2)
        # self.assertEqual(self.n1.count_params(), 800)
        self.assertEqual(len(self.n1._info[0].layer.all_weights), 2)
        self.assertEqual(self.n1.get_shape().as_list()[1:], [10, 10, 10, 32])

    def test_layer_n2(self):

        # self.assertEqual(len(self.n2.all_layers), 3)
        # self.assertEqual(len(self.n2.all_params), 4)
        # self.assertEqual(self.n2.count_params(), 33696)
        self.assertEqual(len(self.n2._info[0].layer.all_weights), 2)
        self.assertEqual(self.n2.get_shape().as_list()[1:], [20, 20, 20, 128])

    def test_layer_n3(self):

        # self.assertEqual(len(self.n3.all_layers), 4)
        # self.assertEqual(len(self.n3.all_params), 6)
        # self.assertEqual(self.n3.count_params(), 144320)
        self.assertEqual(len(self.n3._info[0].layer.all_weights), 1)  # b_init is None
        self.assertEqual(self.n3.get_shape().as_list()[1:], [7, 7, 7, 64])

    def test_layer_n4(self):

        # self.assertEqual(len(self.n3.all_layers), 4)
        # self.assertEqual(len(self.n3.all_params), 6)
        # self.assertEqual(self.n3.count_params(), 144320)
        self.assertEqual(len(self.n4._info[0].layer.all_weights), 2)
        self.assertEqual(self.n4.get_shape().as_list()[1:], [14, 14, 14, 32])


# class Layer_DeformableConvolution_Test(CustomTestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         cls.batch_size = 5
#         cls.inputs_shape = [cls.batch_size, 299, 299, 3]
#         cls.input_layer = Input(cls.inputs_shape, name='input_layer')
#
#         offset1 = tl.layers.Conv2d(
#             18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset1'
#         )(cls.input_layer)
#         cls.net1 = tl.layers.DeformableConv2d(
#             offset1, 32, (3, 3), act=tf.nn.relu, name='deformable1'
#         )(cls.input_layer)
#
#         offset2 = tl.layers.Conv2d(
#             18, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='offset2'
#         )(cls.net1)
#         cls.net2 = tl.layers.DeformableConv2d(
#             offset2, 64, (3, 3), act=tf.nn.relu, name='deformable2'
#         )(cls.net1)
#
#     @classmethod
#     def tearDownClass(cls):
#         pass
#
#     def test_layer_n1(self):
#
#         self.assertEqual(len(self.net1.all_layers), 2)
#         self.assertEqual(len(self.net1.all_params), 2)
#         self.assertEqual(self.net1.count_params(), 896)
#         self.assertEqual(self.net1.outputs.get_shape().as_list()[1:], [299, 299, 32])
#
#     def test_layer_n2(self):
#
#         self.assertEqual(len(self.net2.all_layers), 3)
#         self.assertEqual(len(self.net2.all_params), 4)
#         self.assertEqual(self.net2.count_params(), 19392)
#         self.assertEqual(self.net2.outputs.get_shape().as_list()[1:], [299, 299, 64])

if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
