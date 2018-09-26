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

        ### Fixed length RNNs
        with tf.variable_scope("test_scope"):
            n_step = 8
            cls.model = tl.networks.Sequential(name="My_Sequential_RNN")
            cls.model.add(tl.layers.EmbeddingInputlayer(vocabulary_size=100, embedding_size=50, name='embedding'))
            cls.model.add(tl.layers.RNNLayer(n_hidden=100, n_steps=n_step, return_last=False, name='rnn'))
            cls.model.add(
                tl.layers.BiRNNLayer(n_hidden=50, dropout=0.5, n_steps=n_step, return_last=False, name='birnn')
            )
            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, n_step, 5, 5, 4], name='reshape'))
            cls.model.add(
                tl.layers.ConvLSTMLayer(
                    cell_shape=(5, 5),
                    feature_map=5,
                    filter_size=(3, 3),
                    n_steps=n_step,
                    return_last=False,
                    name='convlstm'
                )
            )
            plh = tf.placeholder(tf.int32, (100, n_step), name='fixed_length_text')

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

        ### Dynamic length RNNs
        with tf.variable_scope("test_scope2"):
            plh = tf.placeholder(tf.int32, (100, None), name='dynamic_length_text')
            cls.model2 = tl.networks.Sequential(name="My_Sequential_DynamicRNN")
            cls.model2.add(tl.layers.EmbeddingInputlayer(vocabulary_size=100, embedding_size=50, name='embedding'))
            cls.model2.add(
                tl.layers.DynamicRNNLayer(
                    cell_fn=tf.contrib.rnn.BasicLSTMCell,
                    n_hidden=20,
                    dropout=0.7,
                    n_layer=2,
                    sequence_length=tl.layers.retrieve_seq_length_op2(plh),
                    return_last=False,  # for encoder, set to True
                    return_seq_2d=False,
                    name='dynamicrnn'
                )
            )
            cls.model2.add(
                tl.layers.BiDynamicRNNLayer(
                    cell_fn=tf.contrib.rnn.BasicLSTMCell,
                    n_hidden=30,
                    dropout=0.9,
                    n_layer=2,
                    sequence_length=tl.layers.retrieve_seq_length_op2(plh),
                    return_last=False,
                    return_seq_2d=False,
                    name='bidynamicrnn'
                )
            )
            cls.train_model2 = cls.model2.compile(plh, reuse=False, is_train=True)
            cls.test_model2 = cls.model2.compile(plh, reuse=True, is_train=False)

        ### Seq2Seq
        with tf.variable_scope("test_scope3"):
            # https://github.com/tensorlayer/seq2seq-chatbot/blob/master/main.py
            # Training Data Placeholders
            batch_size = 10
            encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
            decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
            target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
            # target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask")

            cls.model3 = tl.networks.Sequential(name="My_Sequential_Seq2Seq")
            cls.model3.add(tl.layers.EmbeddingInputlayer(vocabulary_size=100, embedding_size=50, name='embedding'))

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.test_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

        self.assertIsInstance(self.train_model2, tl.models.CompiledNetwork)
        self.assertIsInstance(self.test_model2, tl.models.CompiledNetwork)
        self.assertIsInstance(self.model2, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        print(self.train_model.count_weights())
        self.assertEqual(self.train_model.count_weights(), 127440)
        self.assertEqual(self.test_model.count_weights(), 127440)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 127440)

        self.assertEqual(self.train_model2.count_weights(), 48040)
        self.assertEqual(self.test_model2.count_weights(), 48040)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model2.count_weights(), 48040)

    def test_count_weights_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 1 + 2 + 4 + 2)
        self.assertEqual(len(self.test_model.get_all_weights()), 1 + 2 + 4 + 2)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 1 + 2 + 4 + 2)

        self.assertEqual(len(self.train_model2.get_all_weights()), 13)
        self.assertEqual(len(self.test_model2.get_all_weights()), 13)
        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model2.get_all_weights()), 13)

    def test_count_layers(self):
        ## fixed length RNNs
        self.assertEqual(self.train_model.count_layers(), 6)
        self.assertEqual(self.test_model.count_layers(), 6)
        self.assertEqual(self.model.count_layers(), 6)
        ## dynamic length RNNs
        self.assertEqual(self.train_model2.count_layers(), 4)
        self.assertEqual(self.test_model2.count_layers(), 4)
        self.assertEqual(self.model2.count_layers(), 4)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):
            ## fixed length RNNs
            self.assertEqual(self.train_model["embedding"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["embedding"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["rnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["rnn"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["birnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["birnn"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["convlstm"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["convlstm"].outputs.dtype, tf.float32)

            ## dynamic length RNNs
            self.assertEqual(self.train_model2["embedding"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model2["embedding"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model2["dynamicrnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model2["dynamicrnn"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model2["bidynamicrnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model2["bidynamicrnn"].outputs.dtype, tf.float32)
            # for layer_name in self.train_model.all_layers:
            #
            #     if self.train_model[layer_name].outputs.dtype != tf.float32:
            #         raise RuntimeError(
            #             "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
            #             (layer_name, self.train_model[layer_name].outputs.dtype, tf.float16)
            #         )
            #
            #     if self.test_model[layer_name].outputs.dtype != tf.float32:
            #         raise RuntimeError(
            #             "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
            #             (layer_name, self.test_model[layer_name].outputs.dtype, tf.float16)
            #         )

    def test_network_shapes(self):
        ## fixed length RNNs
        self.assertEqual(self.train_model["embedding"].outputs.shape, (100, 8, 50))
        self.assertEqual(self.test_model["embedding"].outputs.shape, (100, 8, 50))
        self.assertEqual(self.train_model["rnn"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.test_model["rnn"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.train_model["birnn"].outputs.shape, (100, 8, 50 * 2))  # birnn x 2
        self.assertEqual(self.test_model["birnn"].outputs.shape, (100, 8, 50 * 2))
        self.assertEqual(self.train_model["convlstm"].outputs.shape, (100, 8, 5, 5, 5))
        self.assertEqual(self.test_model["convlstm"].outputs.shape, (100, 8, 5, 5, 5))

        ## dynamic RNNs
        tensor_shape = [k._value for k in self.train_model2["embedding"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 50])
        tensor_shape = [k._value for k in self.test_model2["embedding"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 50])
        tensor_shape = [k._value for k in self.train_model2["dynamicrnn"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 20])
        tensor_shape = [k._value for k in self.test_model2["dynamicrnn"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 20])
        tensor_shape = [k._value for k in self.train_model2["bidynamicrnn"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 30 * 2])
        tensor_shape = [k._value for k in self.test_model2["bidynamicrnn"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 30 * 2])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
