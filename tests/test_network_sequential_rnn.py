#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Network_Sequential_RNN_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            n_step = 8

            cls.model = tl.networks.Sequential(name="My_Sequential_RNN")

            cls.model.add(tl.layers.EmbeddingInput(vocabulary_size=100, embedding_size=50, name='embedding_layer_1'))
            cls.model.add(tl.layers.RNN(n_hidden=100, n_steps=n_step, return_last=False, name='rnn_layer_2'))
            cls.model.add(
                tl.layers.BiRNN(n_hidden=50, dropout=0.5, n_steps=n_step, return_last=False, name='birnn_layer_3')
            )
            cls.model.add(tl.layers.Reshape(shape=(-1, n_step, 5, 5, 4), name='reshape_layer_4'))
            cls.model.add(
                tl.layers.ConvLSTM(
                    cell_shape=(5, 5),
                    feature_map=5,
                    filter_size=(3, 3),
                    n_steps=n_step,
                    return_last=False,
                    name='conv_lstm_layer_5'
                )
            )

            plh = tf.placeholder(tf.int32, (100, n_step), name='fixed_length_text')

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 127440)
        self.assertEqual(self.test_model.count_weights(), 127440)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 127440)

    def test_count_weights_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 1 + 2 + 4 + 2)
        self.assertEqual(len(self.test_model.get_all_weights()), 1 + 2 + 4 + 2)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 1 + 2 + 4 + 2)

    def test_count_layers(self):

        self.assertEqual(self.train_model.count_layers(), 6)
        self.assertEqual(self.test_model.count_layers(), 6)
        self.assertEqual(self.model.count_layers(), 6)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if layer_name in ["input_layer"]:
                    target_dtype = tf.int32
                else:
                    target_dtype = tf.float32

                if self.train_model[layer_name].outputs.dtype != target_dtype:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, target_dtype)
                    )

                if self.test_model[layer_name].outputs.dtype != target_dtype:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, target_dtype)
                    )

    def test_layer_local_weights(self):

        # for layer_name in self.train_model.all_layers:
        #    print("self.assertEqual(self.train_model['%s'].count_local_weights(), %d)" % (layer_name, self.train_model[layer_name].count_local_weights()))
        #    print("self.assertEqual(self.test_model['%s'].count_local_weights(), %d)" % (layer_name, self.test_model[layer_name].count_local_weights()))
        #    print()

        self.assertEqual(self.train_model['input_layer'].count_local_weights(), 0)
        self.assertEqual(self.test_model['input_layer'].count_local_weights(), 0)

        self.assertEqual(self.train_model['embedding_layer_1'].count_local_weights(), 5000)
        self.assertEqual(self.test_model['embedding_layer_1'].count_local_weights(), 5000)

        self.assertEqual(self.train_model['rnn_layer_2'].count_local_weights(), 60400)
        self.assertEqual(self.test_model['rnn_layer_2'].count_local_weights(), 60400)

        self.assertEqual(self.train_model['birnn_layer_3'].count_local_weights(), 60400)
        self.assertEqual(self.test_model['birnn_layer_3'].count_local_weights(), 60400)

        self.assertEqual(self.train_model['reshape_layer_4'].count_local_weights(), 0)
        self.assertEqual(self.test_model['reshape_layer_4'].count_local_weights(), 0)

        self.assertEqual(self.train_model['conv_lstm_layer_5'].count_local_weights(), 1640)
        self.assertEqual(self.test_model['conv_lstm_layer_5'].count_local_weights(), 1640)

    def test_network_shapes(self):

        self.assertEqual(self.train_model["embedding_layer_1"].outputs.shape, (100, 8, 50))
        self.assertEqual(self.test_model["embedding_layer_1"].outputs.shape, (100, 8, 50))

        self.assertEqual(self.train_model["rnn_layer_2"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.test_model["rnn_layer_2"].outputs.shape, (100, 8, 100))

        self.assertEqual(self.train_model["birnn_layer_3"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.test_model["birnn_layer_3"].outputs.shape, (100, 8, 100))

        self.assertEqual(self.train_model["reshape_layer_4"].outputs.shape, (100, 8, 5, 5, 4))
        self.assertEqual(self.test_model["reshape_layer_4"].outputs.shape, (100, 8, 5, 5, 4))

        self.assertEqual(self.train_model["conv_lstm_layer_5"].outputs.shape, (100, 8, 5, 5, 5))
        self.assertEqual(self.test_model["conv_lstm_layer_5"].outputs.shape, (100, 8, 5, 5, 5))


class Network_Sequential_Dynamic_RNN_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope2"):

            plh = tf.placeholder(tf.int32, (100, None), name='dynamic_length_text')

            cls.model = tl.networks.Sequential(name="My_Sequential_DynamicRNN")

            cls.model.add(tl.layers.EmbeddingInput(vocabulary_size=100, embedding_size=50, name='embedding_layer_1'))
            cls.model.add(
                tl.layers.DynamicRNN(
                    cell_fn=tf.contrib.rnn.BasicLSTMCell,
                    n_hidden=20,
                    dropout=0.7,
                    n_layer=2,
                    sequence_length=tl.layers.retrieve_seq_length_op2(plh),
                    return_last=False,  # for encoder, set to True
                    return_seq_2d=False,
                    name='dynamic_rnn_layer_2'
                )
            )
            cls.model.add(
                tl.layers.BiDynamicRNN(
                    cell_fn=tf.contrib.rnn.BasicLSTMCell,
                    n_hidden=30,
                    dropout=0.9,
                    n_layer=2,
                    sequence_length=tl.layers.retrieve_seq_length_op2(plh),
                    return_last=False,
                    return_seq_2d=False,
                    name='bidynamic_rnn_layer_3'
                )
            )

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):

        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):

        self.assertEqual(self.train_model.count_weights(), 48040)
        self.assertEqual(self.test_model.count_weights(), 48040)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 48040)

    def test_count_weights_tensors(self):

        self.assertEqual(len(self.train_model.get_all_weights()), 13)
        self.assertEqual(len(self.test_model.get_all_weights()), 13)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 13)

    def test_count_layers(self):

        self.assertEqual(self.train_model.count_layers(), 4)
        self.assertEqual(self.test_model.count_layers(), 4)
        self.assertEqual(self.model.count_layers(), 4)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if layer_name in ["input_layer"]:
                    target_dtype = tf.int32
                else:
                    target_dtype = tf.float32

                if self.train_model[layer_name].outputs.dtype != target_dtype:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, target_dtype)
                    )

                if self.test_model[layer_name].outputs.dtype != target_dtype:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, target_dtype)
                    )

    def test_network_shapes(self):

        tensor_shape = [k._value for k in self.train_model["embedding_layer_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 50])

        tensor_shape = [k._value for k in self.test_model["embedding_layer_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 50])

        tensor_shape = [k._value for k in self.train_model["dynamic_rnn_layer_2"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 20])

        tensor_shape = [k._value for k in self.test_model["dynamic_rnn_layer_2"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 20])

        tensor_shape = [k._value for k in self.train_model["bidynamic_rnn_layer_3"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 30 * 2])

        tensor_shape = [k._value for k in self.test_model["bidynamic_rnn_layer_3"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 30 * 2])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
