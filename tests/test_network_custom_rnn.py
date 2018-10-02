#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_Seq2Seq_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        batch_size = 100

        encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
        decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")

        class MyCustomNetwork(tl.networks.CustomModel):

            def model(self):

                src_vocab_size = 8002
                emb_dim = 1024

                with tf.variable_scope("input_embedding"):

                    net_encode = tl.layers.EmbeddingInputlayer(
                        vocabulary_size=src_vocab_size, embedding_size=emb_dim, name='seq_embedding_layer_1'
                    )

                    net_decode = tl.layers.EmbeddingInputlayer(
                        vocabulary_size=src_vocab_size,
                        embedding_size=emb_dim,
                        reuse_variable_scope=True,
                        name='seq_embedding_layer_1'
                    )

                net_rnn = tl.layers.Seq2Seq(
                    cell_fn=tf.nn.rnn_cell.LSTMCell,
                    n_hidden=emb_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    encode_sequence_length=tl.layers.utils.retrieve_seq_length_op2(encode_seqs),
                    decode_sequence_length=tl.layers.utils.retrieve_seq_length_op2(decode_seqs),
                    dropout=0.5,
                    n_layer=3,
                    return_seq_2d=True,
                    name='seq2seq_layer_2'
                )(net_encode, net_decode)

                net_out = tl.layers.DenseLayer(n_units=src_vocab_size, name='dense_layer_3')(net_rnn)

                return (net_encode, net_decode), net_out

        cls.model = MyCustomNetwork(name="my_custom_network")

        cls.train_model = cls.model.build((encode_seqs, decode_seqs), reuse=False, is_train=True)

        cls.test_model = cls.model.build((encode_seqs, decode_seqs), reuse=True, is_train=False)

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
        self.assertEqual(self.train_model.count_weights(), 74946370)
        self.assertEqual(self.test_model.count_weights(), 74946370)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 74946370)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 15)
        self.assertEqual(len(self.test_model.get_all_weights()), 15)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 15)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 4)
        self.assertEqual(self.test_model.count_layers(), 4)
        self.assertEqual(self.model.count_layers(), 4)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float32)
                    )

                if self.test_model[layer_name].outputs.dtype != tf.float32:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float32)
                    )

    def test_network_shapes(self):

        tensor_shape = [k._value for k in self.train_model["input_embedding/seq_embedding_layer_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 1024])

        tensor_shape = [k._value for k in self.test_model["input_embedding/seq_embedding_layer_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 1024])

        tensor_shape = [k._value for k in self.train_model["input_embedding/seq_embedding_layer_1_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 1024])

        tensor_shape = [k._value for k in self.test_model["input_embedding/seq_embedding_layer_1_1"].outputs.shape]
        self.assertEqual(tensor_shape, [100, None, 1024])

        tensor_shape = [k._value for k in self.train_model["seq2seq_layer_2"].outputs.shape]
        self.assertEqual(tensor_shape, [None, 1024])

        tensor_shape = [k._value for k in self.test_model["seq2seq_layer_2"].outputs.shape]
        self.assertEqual(tensor_shape, [None, 1024])

        tensor_shape = [k._value for k in self.train_model["dense_layer_3"].outputs.shape]
        self.assertEqual(tensor_shape, [None, 8002])

        tensor_shape = [k._value for k in self.test_model["dense_layer_3"].outputs.shape]
        self.assertEqual(tensor_shape, [None, 8002])


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
