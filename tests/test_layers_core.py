#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Core_Helpers_Test(CustomTestCase):

    def test_LayersConfig(self):
        with self.assertRaises(TypeError):
            tl.layers.LayersConfig()

        self.assertIsInstance(tl.layers.LayersConfig.tf_dtype, tf.DType)
        self.assertIsInstance(tl.layers.LayersConfig.set_keep, dict)


class Layer_Core_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.batch_size = 8

        # ============== DenseLayer ==============

        x1 = tf.placeholder(tf.float32, shape=[None, 30])
        net1 = tl.layers.InputLayer(x1, name='input')
        net1 = tl.layers.DenseLayer(net1, n_units=10, name='dense')

        net1.print_layers()
        net1.print_params(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_params = net1.all_params
        cls.net1_n_params = net1.count_params()

        # ============== OneHotInputLayer ==============

        x2 = tf.placeholder(tf.int32, shape=[None])
        net2 = tl.layers.OneHotInputLayer(x2, depth=8, name='onehot')

        net2.print_layers()
        net2.print_params(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_params = net2.all_params
        cls.net2_n_params = net2.count_params()

        # ============== Word2vecEmbeddingInputlayer ==============

        train_inputs = tf.placeholder(tf.int32, shape=cls.batch_size)
        train_labels = tf.placeholder(tf.int32, shape=(cls.batch_size, 1))
        net3 = tl.layers.Word2vecEmbeddingInputlayer(
            inputs=train_inputs, train_labels=train_labels, vocabulary_size=1000, embedding_size=200, num_sampled=64,
            name='word2vec'
        )

        net3.print_layers()
        net3.print_params(False)

        cls.net3_shape = net3.outputs.get_shape().as_list()
        cls.net3_layers = net3.all_layers
        cls.net3_params = net3.all_params
        cls.net3_n_params = net3.count_params()

        # ============== EmbeddingInputlayer ==============

        x4 = tf.placeholder(tf.int32, shape=(cls.batch_size, ))
        net4 = tl.layers.EmbeddingInputlayer(inputs=x4, vocabulary_size=1000, embedding_size=50, name='embed')

        net4.print_layers()
        net4.print_params(False)

        cls.net4_shape = net4.outputs.get_shape().as_list()
        cls.net4_layers = net4.all_layers
        cls.net4_params = net4.all_params
        cls.net4_n_params = net4.count_params()

        # ============== AverageEmbeddingInputlayer ==============

        length = 5
        x5 = tf.placeholder(tf.int32, shape=(cls.batch_size, length))
        net5 = tl.layers.AverageEmbeddingInputlayer(inputs=x5, vocabulary_size=1000, embedding_size=50, name='avg')

        net5.print_layers()
        net5.print_params(False)

        cls.net5_shape = net5.outputs.get_shape().as_list()
        cls.net5_layers = net5.all_layers
        cls.net5_params = net5.all_params
        cls.net5_n_params = net5.count_params()

        # ============== ReconLayer ==============

        x6 = tf.placeholder(tf.float32, shape=(None, 784))
        net6 = tl.layers.InputLayer(x6, name='input')
        net6 = tl.layers.DenseLayer(net6, n_units=196, act=tf.nn.sigmoid, name='dense2')
        net6 = tl.layers.ReconLayer(net6, x_recon=x6, n_units=784, act=tf.nn.sigmoid, name='recon')

        # sess = tf.InteractiveSession()
        # tl.layers.initialize_global_variables(sess)
        # X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
        # net.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1, batch_size=128, print_freq=1, save=True, save_name='w1pre_')

        net6.print_layers()
        net6.print_params(False)

        cls.net6_shape = net6.outputs.get_shape().as_list()
        cls.net6_layers = net6.all_layers
        cls.net6_params = net6.all_params
        cls.net6_n_params = net6.count_params()

        # ============== GaussianNoiseLayer ==============

        x7 = tf.placeholder(tf.float32, shape=(64, 784))
        net7 = tl.layers.InputLayer(x7, name='input')
        net7 = tl.layers.DenseLayer(net7, n_units=100, act=tf.nn.relu, name='dense3')
        net7 = tl.layers.GaussianNoiseLayer(net7, name='gaussian')

        net7.print_layers()
        net7.print_params(False)

        cls.net7_shape = net7.outputs.get_shape().as_list()
        cls.net7_layers = net7.all_layers
        cls.net7_params = net7.all_params
        cls.net7_n_params = net7.count_params()

        # ============== DropconnectDenseLayer ==============

        x8 = tf.placeholder(tf.float32, shape=(64, 784))
        net8 = tl.layers.InputLayer(x8, name='input')
        net8 = tl.layers.DenseLayer(net8, n_units=100, act=tf.nn.relu, name='dense4')
        net8 = tl.layers.DropconnectDenseLayer(net8, keep=0.8, name='dropconnect')

        net8.print_layers()
        net8.print_params(False)

        cls.net8_shape = net8.outputs.get_shape().as_list()
        cls.net8_layers = net8.all_layers
        cls.net8_params = net8.all_params
        cls.net8_n_params = net8.count_params()

        # ============== QuanDenseLayer ==============

        x9 = tf.placeholder(tf.float32, shape=(None, 30))
        net9 = tl.layers.InputLayer(x9, name='input')
        net9 = tl.layers.QuanDenseLayer(net9, n_units=10, act=tf.nn.relu, name='quandense')

        net9.print_layers()
        net9.print_params(False)

        cls.net9_shape = net9.outputs.get_shape().as_list()
        cls.net9_layers = net9.all_layers
        cls.net9_params = net9.all_params
        cls.net9_n_params = net9.count_params()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(self.net1_shape[-1], 10)
        self.assertEqual(len(self.net1_layers), 2)
        self.assertEqual(len(self.net1_params), 2)
        self.assertEqual(self.net1_n_params, 310)

    def test_net2(self):
        self.assertEqual(self.net2_shape[-1], 8)
        self.assertEqual(len(self.net2_layers), 1)
        self.assertEqual(len(self.net2_params), 0)
        self.assertEqual(self.net2_n_params, 0)

    def test_net3(self):
        self.assertEqual(self.net3_shape, [self.batch_size, 200])
        self.assertEqual(len(self.net3_layers), 1)
        self.assertEqual(len(self.net3_params), 3)
        self.assertEqual(self.net3_n_params, 401000)

    def test_net4(self):
        self.assertEqual(self.net4_shape, [self.batch_size, 50])
        self.assertEqual(len(self.net4_layers), 1)
        self.assertEqual(len(self.net4_params), 1)
        self.assertEqual(self.net4_n_params, 50000)

    def test_net5(self):
        self.assertEqual(self.net5_shape, [self.batch_size, 50])
        self.assertEqual(len(self.net5_layers), 1)
        self.assertEqual(len(self.net5_params), 1)
        self.assertEqual(self.net5_n_params, 50000)

    def test_net6(self):
        self.assertEqual(self.net6_shape[-1], 784)
        self.assertEqual(len(self.net6_layers), 3)
        self.assertEqual(len(self.net6_params), 4)
        self.assertEqual(self.net6_n_params, 308308)

    def test_net7(self):
        self.assertEqual(self.net7_shape, [64, 100])
        self.assertEqual(len(self.net7_layers), 3)
        self.assertEqual(len(self.net7_params), 2)
        self.assertEqual(self.net7_n_params, 78500)

    def test_net8(self):
        self.assertEqual(self.net8_shape, [64, 100])
        self.assertEqual(len(self.net8_layers), 3)
        self.assertEqual(len(self.net8_params), 4)
        self.assertEqual(self.net8_n_params, 88600)

    def test_net9(self):
        self.assertEqual(self.net9_shape[-1], 10)
        self.assertEqual(len(self.net9_layers), 2)
        self.assertEqual(len(self.net9_params), 2)
        self.assertEqual(self.net9_n_params, 310)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
