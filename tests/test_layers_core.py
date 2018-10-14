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

        # ============== Dense ==============

        x1 = tf.placeholder(tf.float32, shape=[None, 30])
        net1 = tl.layers.Input(name='input')(x1)
        net1 = tl.layers.Dense(n_units=10, name='dense')(net1)

        net1.print_layers()
        net1.print_weights(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_weights = net1.all_weights
        cls.net1_n_weights = net1.count_weights()

        # ============== OneHotInput ==============

        x2 = tf.placeholder(tf.int32, shape=[None])
        net2 = tl.layers.OneHotInput(x2, depth=8, name='onehot')

        net2.print_layers()
        net2.print_weights(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_weights = net2.all_weights
        cls.net2_n_weights = net2.count_weights()

        # ============== Word2vecEmbeddingInput ==============

        train_inputs = tf.placeholder(tf.int32, shape=cls.batch_size)
        train_labels = tf.placeholder(tf.int32, shape=(cls.batch_size, 1))
        net3 = tl.layers.Word2vecEmbeddingInput(
            train_labels=train_labels, vocabulary_size=1000, embedding_size=200, num_sampled=64, name='word2vec'
        )(train_inputs)

        net3.print_layers()
        net3.print_weights(False)

        cls.net3_shape = net3.outputs.get_shape().as_list()
        cls.net3_layers = net3.all_layers
        cls.net3_weights = net3.all_weights
        cls.net3_n_weights = net3.count_weights()

        # ============== EmbeddingInput ==============

        x4 = tf.placeholder(tf.int32, shape=(cls.batch_size, ))
        net4 = tl.layers.EmbeddingInput(vocabulary_size=1000, embedding_size=50, name='embed')(x4)

        net4.print_layers()
        net4.print_weights(False)

        cls.net4_shape = net4.outputs.get_shape().as_list()
        cls.net4_layers = net4.all_layers
        cls.net4_weights = net4.all_weights
        cls.net4_n_weights = net4.count_weights()

        # ============== AverageEmbeddingInput ==============

        length = 5
        x5 = tf.placeholder(tf.int32, shape=(cls.batch_size, length))
        net5 = tl.layers.AverageEmbeddingInput(vocabulary_size=1000, embedding_size=50, name='avg')(x5)

        net5.print_layers()
        net5.print_weights(False)

        cls.net5_shape = net5.outputs.get_shape().as_list()
        cls.net5_layers = net5.all_layers
        cls.net5_weights = net5.all_weights
        cls.net5_n_weights = net5.count_weights()

        # ============== ReconLayer ==============

        x6 = tf.placeholder(tf.float32, shape=(None, 784))
        net6 = tl.layers.Input(name='input')(x6)
        net6 = tl.layers.Dense(n_units=196, act=tf.nn.sigmoid, name='dense2')(net6)
        net6 = tl.layers.ReconLayer(x_recon=x6, n_units=784, act=tf.nn.sigmoid, name='recon')(net6)

        # sess = tf.InteractiveSession()
        # tl.layers.initialize_global_variables(sess)
        # X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
        # net.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1, batch_size=128, print_freq=1, save=True, save_name='w1pre_')

        net6.print_layers()
        net6.print_weights(False)

        cls.net6_shape = net6.outputs.get_shape().as_list()
        cls.net6_layers = net6.all_layers
        cls.net6_weights = net6.all_weights
        cls.net6_n_weights = net6.count_weights()

        # ============== GaussianNoiseLayer ==============

        x7 = tf.placeholder(tf.float32, shape=(64, 784))
        net7 = tl.layers.Input(name='input')(x7)
        net7 = tl.layers.Dense(n_units=100, act=tf.nn.relu, name='dense3')(net7)
        net7 = tl.layers.GaussianNoise(name='gaussian')(net7)

        net7.print_layers()
        net7.print_weights(False)

        cls.net7_shape = net7.outputs.get_shape().as_list()
        cls.net7_layers = net7.all_layers
        cls.net7_weights = net7.all_weights
        cls.net7_n_weights = net7.count_weights()

        # ============== DropconnectDense ==============

        x8 = tf.placeholder(tf.float32, shape=(64, 784))
        net8 = tl.layers.Input(name='input')(x8)
        net8 = tl.layers.Dense(n_units=100, act=tf.nn.relu, name='dense4')(net8)
        net8 = tl.layers.DropconnectDense(keep=0.8, name='dropconnect')(net8)

        net8.print_layers()
        net8.print_weights(False)

        cls.net8_shape = net8.outputs.get_shape().as_list()
        cls.net8_layers = net8.all_layers
        cls.net8_weights = net8.all_weights
        cls.net8_n_weights = net8.count_weights()

        # ============== QuantizedDense ==============

        x9 = tf.placeholder(tf.float32, shape=(None, 30))
        net9 = tl.layers.Input(name='input')(x9)
        net9 = tl.layers.QuantizedDense(n_units=10, act=tf.nn.relu, name='quandense')(net9)

        net9.print_layers()
        net9.print_weights(False)

        cls.net9_shape = net9.outputs.get_shape().as_list()
        cls.net9_layers = net9.all_layers
        cls.net9_weights = net9.all_weights
        cls.net9_n_weights = net9.count_weights()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(self.net1_shape[-1], 10)
        self.assertEqual(len(self.net1_layers), 2)
        self.assertEqual(len(self.net1_weights), 2)
        self.assertEqual(self.net1_n_weights, 310)

    def test_net2(self):
        self.assertEqual(self.net2_shape[-1], 8)
        self.assertEqual(len(self.net2_layers), 1)
        self.assertEqual(len(self.net2_weights), 0)
        self.assertEqual(self.net2_n_weights, 0)

    def test_net3(self):
        self.assertEqual(self.net3_shape, [self.batch_size, 200])
        self.assertEqual(len(self.net3_layers), 1)
        self.assertEqual(len(self.net3_weights), 3)
        self.assertEqual(self.net3_n_weights, 401000)

    def test_net4(self):
        self.assertEqual(self.net4_shape, [self.batch_size, 50])
        self.assertEqual(len(self.net4_layers), 1)
        self.assertEqual(len(self.net4_weights), 1)
        self.assertEqual(self.net4_n_weights, 50000)

    def test_net5(self):
        self.assertEqual(self.net5_shape, [self.batch_size, 50])
        self.assertEqual(len(self.net5_layers), 1)
        self.assertEqual(len(self.net5_weights), 1)
        self.assertEqual(self.net5_n_weights, 50000)

    def test_net6(self):
        self.assertEqual(self.net6_shape[-1], 784)
        self.assertEqual(len(self.net6_layers), 3)
        self.assertEqual(len(self.net6_weights), 4)
        self.assertEqual(self.net6_n_weights, 308308)

    def test_net7(self):
        self.assertEqual(self.net7_shape, [64, 100])
        self.assertEqual(len(self.net7_layers), 3)
        self.assertEqual(len(self.net7_weights), 2)
        self.assertEqual(self.net7_n_weights, 78500)

    def test_net8(self):
        self.assertEqual(self.net8_shape, [64, 100])
        self.assertEqual(len(self.net8_layers), 3)
        self.assertEqual(len(self.net8_weights), 4)
        self.assertEqual(self.net8_n_weights, 88600)

    def test_net9(self):
        self.assertEqual(self.net9_shape[-1], 10)
        self.assertEqual(len(self.net9_layers), 2)
        self.assertEqual(len(self.net9_weights), 2)
        self.assertEqual(self.net9_n_weights, 310)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
