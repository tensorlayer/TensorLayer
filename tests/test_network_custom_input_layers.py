#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_AverageEmbeddingInputlayer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.vocab_size = 100000
        cls.embedding_size = 50
        cls.n_labels = 2

        with tf.variable_scope("test_scope_1"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):

                    # Network structure
                    input_layer = tl.layers.AverageEmbeddingInputlayer(
                        vocabulary_size=cls.vocab_size, embedding_size=cls.embedding_size, name="input_avg_emb_layer_1"
                    )

                    network = tl.layers.DenseLayer(n_units=cls.n_labels, name="dense_layer_2")(input_layer)

                    return input_layer, network

            cls.model = MyCustomNetwork(name="my_custom_network_1")

            plh = tf.placeholder(tf.int32, shape=[100, None], name='inputs')

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 5000102)
        self.assertEqual(self.test_model.count_weights(), 5000102)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 5000102)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 3)
        self.assertEqual(len(self.test_model.get_all_weights()), 3)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 3)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 2)
        self.assertEqual(self.test_model.count_layers(), 2)
        self.assertEqual(self.model.count_layers(), 2)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float32)
                    )

            for layer_name in self.test_model.all_layers:

                if self.test_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float32)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_avg_emb_layer_1"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["input_avg_emb_layer_1"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 2))


class CustomNetwork_EmbeddingInputlayer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.vocab_size = 100000
        cls.embedding_size = 50
        cls.n_labels = 2

        with tf.variable_scope("test_scope_2"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):

                    # Network structure
                    input_layer = tl.layers.EmbeddingInputlayer(
                        vocabulary_size=cls.vocab_size, embedding_size=cls.embedding_size, name="input_emb_layer_1"
                    )

                    network = tl.layers.DenseLayer(n_units=cls.n_labels, name="dense_layer_2")(input_layer)

                    return input_layer, network

            cls.model = MyCustomNetwork(name="my_custom_network_2")

            plh = tf.placeholder(tf.int32, shape=(100, ), name='inputs')

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 5000102)
        self.assertEqual(self.test_model.count_weights(), 5000102)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 5000102)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 3)
        self.assertEqual(len(self.test_model.get_all_weights()), 3)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 3)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 2)
        self.assertEqual(self.test_model.count_layers(), 2)
        self.assertEqual(self.model.count_layers(), 2)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float32)
                    )

            for layer_name in self.test_model.all_layers:

                if self.test_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float32)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_emb_layer_1"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["input_emb_layer_1"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 2))


class CustomNetwork_OneHotInputLayer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.vocab_size = 100000
        cls.embedding_size = 50
        cls.n_labels = 2

        with tf.variable_scope("test_scope_4"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):

                    # Network structure
                    input_layer = tl.layers.OneHotInputLayer(depth=8, name='one_hot_encoding_layer_1')

                    network = tl.layers.DenseLayer(n_units=cls.n_labels, name="dense_layer_2")(input_layer)

                    return input_layer, network

            cls.model = MyCustomNetwork(name="my_custom_network_4")

            data_plh = tf.placeholder(tf.int32, shape=[100])

            cls.train_model = cls.model.build(data_plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(data_plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 18)
        self.assertEqual(self.test_model.count_weights(), 18)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 18)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 2)
        self.assertEqual(len(self.test_model.get_all_weights()), 2)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 2)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 2)
        self.assertEqual(self.test_model.count_layers(), 2)
        self.assertEqual(self.model.count_layers(), 2)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float32)
                    )

            for layer_name in self.test_model.all_layers:

                if self.test_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float32)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["one_hot_encoding_layer_1"].outputs.shape, (100, 8))
        self.assertEqual(self.test_model["one_hot_encoding_layer_1"].outputs.shape, (100, 8))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 2))


class CustomNetwork_Word2vecEmbeddingInputlayer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.vocab_size = 100000
        cls.embedding_size = 50
        cls.n_labels = 2

        with tf.variable_scope("test_scope_3"):

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):

                    # Network structure
                    input_layer = tl.layers.Word2vecEmbeddingInputlayer(
                        vocabulary_size=1000, embedding_size=200, num_sampled=64, name='word2vec_layer_1'
                    )

                    network = tl.layers.DenseLayer(n_units=cls.n_labels, name="dense_layer_2")(input_layer)

                    return input_layer, network

            cls.model = MyCustomNetwork(name="my_custom_network_3")

            train_inputs = tf.placeholder(tf.int32, shape=100)
            train_labels = tf.placeholder(tf.int32, shape=(100, 1))

            cls.train_model = cls.model.build([train_inputs, train_labels], reuse=False, is_train=True)
            cls.test_model = cls.model.build([train_inputs, train_labels], reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 401402)
        self.assertEqual(self.test_model.count_weights(), 401402)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 401402)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 5)
        self.assertEqual(len(self.test_model.get_all_weights()), 5)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 5)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 2)
        self.assertEqual(self.test_model.count_layers(), 2)
        self.assertEqual(self.model.count_layers(), 2)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float32)
                    )

            for layer_name in self.test_model.all_layers:

                if self.test_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float32)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.train_model["word2vec_layer_1"].outputs.shape, (100, 200))
        self.assertEqual(self.test_model["word2vec_layer_1"].outputs.shape, (100, 200))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 2))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
