#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Convolution_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):

        self.batch_size = 8
        self.inputs_shape = [self.batch_size, 100, 1]
        self.input_layer = tl.layers.Input(self.inputs_shape, name='input_layer')

        self.conv1dlayer1 = tl.layers.Conv1d(in_channels=1, n_filter=32, filter_size=5, stride=2)
        self.n1 = self.conv1dlayer1(self.input_layer)

        self.conv1dlayer2 = tl.layers.Conv1d(in_channels=32, n_filter=32, filter_size=5, stride=2)
        self.n2 = self.conv1dlayer2(self.n1)

        self.dconv1dlayer1 = tl.layers.DeConv1d(n_filter=64, in_channels=32, filter_size=5, name='deconv1dlayer')
        self.n3 = self.dconv1dlayer1(self.n2)

        self.separableconv1d1 = tl.layers.SeparableConv1d(in_channels=1, n_filter=16, filter_size=3, stride=2)
        self.n4 = self.separableconv1d1(self.input_layer)

        self.separableconv1d2 = tl.layers.SeparableConv1d(
            in_channels=1, n_filter=16, filter_size=3, stride=2, depth_multiplier=4
        )
        self.n5 = self.separableconv1d2(self.input_layer)

        self.separableconv1d3 = tl.layers.SeparableConv1d(
            in_channels=1, n_filter=16, filter_size=3, stride=2, depth_multiplier=4, b_init=None
        )
        self.n6 = self.separableconv1d3(self.input_layer)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n1(self):
        self.assertEqual(len(self.conv1dlayer1.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n1), [self.batch_size, 50, 32])

    def test_layer_n2(self):
        self.assertEqual(len(self.conv1dlayer2.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n2), [self.batch_size, 25, 32])

    def test_layer_n3(self):
        self.assertEqual(len(self.dconv1dlayer1.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n3), [self.batch_size, 25, 64])

    def test_layer_n4(self):
        self.assertEqual(len(self.separableconv1d1.all_weights), 3)
        self.assertEqual(tl.get_tensor_shape(self.n4), [self.batch_size, 50, 16])

    def test_layer_n5(self):
        self.assertEqual(len(self.separableconv1d2.all_weights), 3)
        self.assertEqual(tl.get_tensor_shape(self.n5), [self.batch_size, 50, 16])

    def test_layer_n6(self):
        self.assertEqual(len(self.separableconv1d3.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n6), [self.batch_size, 50, 16])


class Layer_Convolution_2D_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):

        self.batch_size = 5
        self.inputs_shape = [self.batch_size, 400, 400, 3]
        self.input_layer = tl.layers.Input(self.inputs_shape, name='input_layer')

        self.conv2dlayer1 = tl.layers.Conv2d(
            n_filter=32, in_channels=3, strides=(2, 2), filter_size=(5, 5), padding='SAME',
            b_init=tl.initializers.truncated_normal(0.01), name='conv2dlayer'
        )
        self.n1 = self.conv2dlayer1(self.input_layer)

        self.conv2dlayer2 = tl.layers.Conv2d(
            n_filter=32, in_channels=32, filter_size=(3, 3), strides=(2, 2), act=None, name='conv2d'
        )
        self.n2 = self.conv2dlayer2(self.n1)

        self.conv2dlayer3 = tl.layers.Conv2d(
            in_channels=32, n_filter=32, filter_size=(3, 3), strides=(2, 2), act=tl.ReLU, b_init=None,
            name='conv2d_no_bias'
        )
        self.n3 = self.conv2dlayer3(self.n2)

        self.dconv2dlayer = tl.layers.DeConv2d(
            n_filter=32, in_channels=32, filter_size=(5, 5), strides=(2, 2), name='deconv2dlayer'
        )
        self.n4 = self.dconv2dlayer(self.n3)

        self.dwconv2dlayer = tl.layers.DepthwiseConv2d(
            in_channels=32, filter_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), act=tl.ReLU, depth_multiplier=2,
            name='depthwise'
        )
        self.n5 = self.dwconv2dlayer(self.n4)

        self.separableconv2d = tl.layers.SeparableConv2d(
            in_channels=3, filter_size=(3, 3), strides=(2, 2), dilation_rate=(2, 2), act=tl.ReLU, depth_multiplier=3,
            name='separableconv2d'
        )
        self.n6 = self.separableconv2d(self.input_layer)

        self.groupconv2d = tl.layers.GroupConv2d(
            in_channels=3, n_filter=18, filter_size=(3, 3), strides=(2, 2), dilation_rate=(3, 3), n_group=3,
            act=tl.ReLU, name='groupconv2d'
        )
        self.n7 = self.groupconv2d(self.input_layer)

        self.binaryconv2d = tl.layers.BinaryConv2d(
            in_channels=3, n_filter=32, filter_size=(3, 3), strides=(2, 2), dilation_rate=(2, 2), act=tl.ReLU,
            name='binaryconv2d'
        )
        self.n8 = self.binaryconv2d(self.input_layer)

        self.dorefaconv2d = tl.layers.DorefaConv2d(
            bitA=2, bitW=8, in_channels=3, n_filter=16, filter_size=(3, 3), strides=(2, 2), dilation_rate=(2, 2),
            act=tl.ReLU, name='dorefaconv2d'
        )
        self.n9 = self.dorefaconv2d(self.input_layer)

    @classmethod
    def tearDownClass(cls):
        pass
        # tf.reset_default_graph()

    def test_layer_n1(self):
        self.assertEqual(len(self.conv2dlayer1.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n1), [self.batch_size, 200, 200, 32])

    def test_layer_n2(self):
        self.assertEqual(len(self.conv2dlayer2.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n2), [self.batch_size, 100, 100, 32])

    def test_layer_n3(self):
        self.assertEqual(len(self.conv2dlayer3.all_weights), 1)  # b_init is None
        self.assertEqual(tl.get_tensor_shape(self.n3), [self.batch_size, 50, 50, 32])

    def test_layer_n4(self):
        self.assertEqual(len(self.dconv2dlayer.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n4), [self.batch_size, 100, 100, 32])

    def test_layer_n5(self):
        self.assertEqual(len(self.dwconv2dlayer.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n5), [self.batch_size, 100, 100, 64])

    def test_layer_n6(self):
        self.assertEqual(len(self.separableconv2d.all_weights), 3)
        self.assertEqual(tl.get_tensor_shape(self.n6), [self.batch_size, 198, 198, 32])

    def test_layer_n7(self):
        self.assertEqual(len(self.groupconv2d.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n7), [self.batch_size, 200, 200, 18])

    def test_layer_n8(self):
        self.assertEqual(len(self.binaryconv2d.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n8), [self.batch_size, 198, 198, 32])

    def test_layer_n9(self):
        self.assertEqual(len(self.dorefaconv2d.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n9), [self.batch_size, 200, 200, 16])


class Layer_Convolution_3D_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        print("\n#################################")

        self.batch_size = 5
        self.inputs_shape = [self.batch_size, 20, 20, 20, 3]
        self.input_layer = tl.layers.Input(self.inputs_shape, name='input_layer')

        self.conv3dlayer1 = tl.layers.Conv3d(n_filter=32, in_channels=3, filter_size=(2, 2, 2), strides=(2, 2, 2))
        self.n1 = self.conv3dlayer1(self.input_layer)

        self.deconv3dlayer = tl.layers.DeConv3d(n_filter=128, in_channels=32, filter_size=(2, 2, 2), strides=(2, 2, 2))
        self.n2 = self.deconv3dlayer(self.n1)

        self.conv3dlayer2 = tl.layers.Conv3d(
            n_filter=64, in_channels=128, filter_size=(3, 3, 3), strides=(3, 3, 3), act=tl.ReLU, b_init=None,
            name='conv3d_no_bias'
        )
        self.n3 = self.conv3dlayer2(self.n2)

    @classmethod
    def tearDownClass(self):
        pass

    def test_layer_n1(self):
        self.assertEqual(len(self.conv3dlayer1.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n1), [self.batch_size, 10, 10, 10, 32])

    def test_layer_n2(self):
        self.assertEqual(len(self.deconv3dlayer.all_weights), 2)
        self.assertEqual(tl.get_tensor_shape(self.n2), [self.batch_size, 20, 20, 20, 128])

    def test_layer_n3(self):
        self.assertEqual(len(self.conv3dlayer2.all_weights), 1)  # b_init is None
        self.assertEqual(tl.get_tensor_shape(self.n3), [self.batch_size, 7, 7, 7, 64])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
