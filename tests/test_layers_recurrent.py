#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Recurrent_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.net1_batch_size = 32
        cls.net2_batch_size = 10
        cls.net3_batch_size = 10
        cls.net5_batch_size = 32
        cls.net11_batch_size = 32

        cls.vocab_size = 30
        cls.hidden_size = 20
        cls.image_size = 100
        cls.embedding_size = 20

        cls.num_steps = 5

        cls.keep_prob = 0.8
        cls.is_train = True

        # =============================== RNN encoder ===============================

        input_data = tf.placeholder(tf.int32, [cls.net1_batch_size, cls.num_steps])

        net1 = tl.layers.EmbeddingInput(
            vocabulary_size=cls.vocab_size, embedding_size=cls.hidden_size, name='embedding'
        )(input_data)
        net1 = tl.layers.Dropout(keep=cls.keep_prob, is_fix=True, name='drop1')(net1, is_train=cls.is_train)
        net1 = tl.layers.RNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.hidden_size,
            n_steps=cls.num_steps,
            return_last=False,
            name='lstm1'
        )(net1)

        # lstm1 = net1

        net1 = tl.layers.Dropout(keep=cls.keep_prob, is_fix=True, name='drop2')(net1, is_train=cls.is_train)
        net1 = tl.layers.RNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.hidden_size,
            n_steps=cls.num_steps,
            return_last=True,
            name='lstm2'
        )(net1)

        # lstm2 = net1

        net1 = tl.layers.Dropout(keep=cls.keep_prob, is_fix=True, name='drop3')(net1, is_train=cls.is_train)
        net1 = tl.layers.Dense(n_units=cls.vocab_size, name='output')(net1)

        net1.print_layers()
        net1.print_weights(False)

        cls.net1_shape = net1.outputs.get_shape().as_list()
        cls.net1_layers = net1.all_layers
        cls.net1_weights = net1.all_weights
        cls.net1_n_weights = net1.count_weights()

        # =============================== CNN+RNN encoder ===============================

        x2 = tf.placeholder(tf.float32, shape=[cls.net2_batch_size, cls.image_size, cls.image_size, 1])
        net2 = tl.layers.Input(name='in')(x2)

        net2 = tl.layers.Conv2d(n_filter=32, filter_size=(5, 5), strides=(2, 2), act=tf.nn.relu, name='cnn1')(net2)
        net2 = tl.layers.MaxPool2d(filter_size=(2, 2), strides=(2, 2), name='pool1')(net2)
        net2 = tl.layers.Conv2d(n_filter=10, filter_size=(5, 5), strides=(2, 2), act=tf.nn.relu, name='cnn2')(net2)
        net2 = tl.layers.MaxPool2d(filter_size=(2, 2), strides=(2, 2), name='pool2')(net2)

        net2 = tl.layers.Flatten(name='flatten')(net2)
        net2 = tl.layers.Reshape(shape=(-1, cls.num_steps, int(net2.outputs._shape[-1])))(net2)

        net2 = tl.layers.RNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=200,
            n_steps=cls.num_steps,
            return_last=False,
            return_seq_2d=True,
            name='rnn'
        )(net2)

        net2 = tl.layers.Dense(n_units=3, name='out')(net2)

        net2.print_layers()
        net2.print_weights(False)

        cls.net2_shape = net2.outputs.get_shape().as_list()
        cls.net2_layers = net2.all_layers
        cls.net2_weights = net2.all_weights
        cls.net2_n_weights = net2.count_weights()

        tl.logging.debug("ALL LAYERS: ##################################################", cls.net2_layers)

        # =============================== Bidirectional Synced input and output ===============================

        x3 = tf.placeholder(tf.int32, [cls.net3_batch_size, cls.num_steps])

        net3 = tl.layers.EmbeddingInput(
            vocabulary_size=cls.vocab_size, embedding_size=cls.hidden_size, dtype=tf.float32, name='emb'
        )(x3)
        net3 = tl.layers.BiRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.hidden_size,
            n_steps=cls.num_steps,
            return_last=False,
            return_seq_2d=False,
            name='birnn'
        )(net3)

        net3.print_layers()
        net3.print_weights(False)

        cls.net3_shape = net3.outputs.get_shape().as_list()
        cls.net3_layers = net3.all_layers
        cls.net3_weights = net3.all_weights
        cls.net3_n_weights = net3.count_weights()

        # n_layer=2
        net4 = tl.layers.EmbeddingInput(
            vocabulary_size=cls.vocab_size, embedding_size=cls.hidden_size, name='emb2'
        )(x3)
        net4 = tl.layers.BiRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.hidden_size,
            n_steps=cls.num_steps,
            n_layer=2,
            return_last=False,
            return_seq_2d=False,
            name='birnn2'
        )(net4)

        net4.print_layers()
        net4.print_weights(False)

        cls.net4_shape = net4.outputs.get_shape().as_list()
        cls.net4_layers = net4.all_layers
        cls.net4_weights = net4.all_weights
        cls.net4_n_weights = net4.count_weights()

        ## TODO: ConvLSTM
        # image_size = 100
        # batch_size = 10
        # num_steps = 5
        # x = tf.placeholder(tf.float32, shape=[batch_size, num_steps, image_size, image_size, 3])
        # net = tl.layers.Input(x, name='in2')
        # net = tl.layers.ConvLSTM(net,
        #             feature_map=1,
        #             filter_size=(3, 3),
        #             cell_fn=tl.layers.BasicConvLSTMCell,
        #             initializer=tf.random_uniform_initializer(-0.1, 0.1),
        #             n_steps=num_steps,
        #             initial_state=None,
        #             return_last=False,
        #             return_seq_2d=False,
        #             name='convlstm')

        # =============================== Dynamic Synced input and output ===============================

        input_seqs = tf.placeholder(dtype=tf.int64, shape=[cls.net5_batch_size, None], name="input")
        nin = tl.layers.EmbeddingInput(
            vocabulary_size=cls.vocab_size, embedding_size=cls.embedding_size, name='seq_embedding'
        )(input_seqs)

        rnn = tl.layers.DynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=(cls.keep_prob if cls.is_train else None),
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            return_last=False,
            return_seq_2d=True,
            name='dynamicrnn'
        )(nin)

        net5 = tl.layers.Dense(n_units=cls.vocab_size, name="o")(rnn)

        net5.print_layers()
        net5.print_weights(False)

        cls.net5_shape = net5.outputs.get_shape().as_list()
        cls.net5_rnn_shape = rnn.outputs.get_shape().as_list()
        cls.net5_layers = net5.all_layers
        cls.net5_weights = net5.all_weights
        cls.net5_n_weights = net5.count_weights()

        # n_layer=3
        nin = tl.layers.EmbeddingInput(
            vocabulary_size=cls.vocab_size, embedding_size=cls.embedding_size, name='seq_embedding2'
        )(input_seqs)
        rnn = tl.layers.DynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=(cls.keep_prob if cls.is_train else None),
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            n_layer=3,
            return_last=False,
            return_seq_2d=True,
            name='dynamicrnn2'
        )(nin)

        # net6 = tl.layers.Dense(rnn, n_units=cls.vocab_size, name="o2")

        net6 = tl.layers.DynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=None,
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            n_layer=3,
            return_last=False,
            return_seq_2d=False,
            name='dynamicrnn3'
        )(nin)

        # net6 = tl.layers.Dense(rnn, n_units=vocab_size, name="o3")

        net6.print_layers()
        net6.print_weights(False)

        cls.net6_shape = net6.outputs.get_shape().as_list()
        cls.net6_rnn_shape = rnn.outputs.get_shape().as_list()

        net7 = tl.layers.DynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=None,
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            n_layer=1,
            return_last=True,
            return_seq_2d=False,
            name='dynamicrnn4'
        )(nin)

        net7.print_layers()
        net7.print_weights(False)

        cls.net7_shape = net7.outputs.get_shape().as_list()

        net8 = tl.layers.DynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=None,
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            n_layer=1,
            return_last=True,
            return_seq_2d=True,
            name='dynamicrnn5'
        )(nin)

        net8.print_layers()
        net8.print_weights(False)

        cls.net8_shape = net8.outputs.get_shape().as_list()

        # =============================== BiDynamic Synced input and output ===============================

        rnn = tl.layers.BiDynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=(cls.keep_prob if cls.is_train else None),
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            return_last=False,
            return_seq_2d=True,
            name='bidynamicrnn'
        )(nin)

        net9 = tl.layers.Dense(rnn, n_units=cls.vocab_size, name="o4")

        net9.print_layers()
        net9.print_weights(False)

        cls.net9_shape = net9.outputs.get_shape().as_list()
        cls.net9_rnn_shape = rnn.outputs.get_shape().as_list()
        cls.net9_layers = net9.all_layers
        cls.net9_weights = net9.all_weights
        cls.net9_n_weights = net9.count_weights()

        # n_layer=2
        rnn = tl.layers.BiDynamicRNN(
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=cls.embedding_size,
            dropout=(cls.keep_prob if cls.is_train else None),
            sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
            n_layer=2,
            return_last=False,
            return_seq_2d=True,
            name='bidynamicrnn2'
        )(nin)

        net10 = tl.layers.Dense(n_units=cls.vocab_size, name="o5")(rnn)

        net10.print_layers()
        net10.print_weights(False)

        cls.net10_shape = net10.outputs.get_shape().as_list()
        cls.net10_rnn_shape = rnn.outputs.get_shape().as_list()
        cls.net10_layers = net10.all_layers
        cls.net10_weights = net10.all_weights
        cls.net10_n_weights = net10.count_weights()

        # =============================== Seq2Seq ===============================

        encode_seqs = tf.placeholder(dtype=tf.int64, shape=[cls.net11_batch_size, None], name="encode_seqs")
        decode_seqs = tf.placeholder(dtype=tf.int64, shape=[cls.net11_batch_size, None], name="decode_seqs")
        # target_seqs = tf.placeholder(dtype=tf.int64, shape=[cls.net11_batch_size, None], name="target_seqs")
        # target_mask = tf.placeholder(dtype=tf.int64, shape=[cls.net11_batch_size, None], name="target_mask")  # tl.prepro.sequences_get_mask()

        with tf.variable_scope("model"):
            # for chatbot, you can use the same embedding layer,
            # for translation, you may want to use 2 seperated embedding layers

            with tf.variable_scope("embedding") as vs:
                net_encode = tl.layers.EmbeddingInput(
                    vocabulary_size=10000, embedding_size=200, name='seq_embed'
                )(encode_seqs)
                vs.reuse_variables()
                # tl.layers.set_name_reuse(True)
                net_decode = tl.layers.EmbeddingInput(
                    vocabulary_size=10000, embedding_size=200, name='seq_embed'
                )(decode_seqs)

            net11 = tl.layers.Seq2Seq(
                net_encode,
                net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=200,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op2(decode_seqs),
                initial_state_encode=None,
                dropout=None,
                n_layer=2,
                return_seq_2d=True,
                name='Seq2seq'
            )

        net11 = tl.layers.Dense(n_units=10000, name='oo')(net11)

        # e_loss = tl.cost.cross_entropy_seq_with_mask(logits=net11.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')
        # y = tf.nn.softmax(net11.outputs)

        net11.print_layers()
        net11.print_weights(False)

        cls.net11_shape = net11.outputs.get_shape().as_list()
        cls.net11_layers = net11.all_layers
        cls.net11_weights = net11.all_weights
        cls.net11_n_weights = net11.count_weights()

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_net1(self):
        self.assertEqual(self.net1_shape, [self.net1_batch_size, self.vocab_size])
        self.assertEqual(len(self.net1_layers), 7)
        self.assertEqual(len(self.net1_weights), 7)
        self.assertEqual(self.net1_n_weights, 7790)

    def test_net2(self):
        self.assertEqual(self.net2_shape, [self.net2_batch_size, 3])
        self.assertEqual(len(self.net2_layers), 9)
        self.assertEqual(len(self.net2_weights), 8)
        self.assertEqual(self.net2_n_weights, 562245)

    def test_net3(self):
        self.assertEqual(self.net3_shape[1:3], [self.num_steps, self.hidden_size * 2])
        self.assertEqual(len(self.net3_layers), 2)
        self.assertEqual(len(self.net3_weights), 5)
        self.assertEqual(self.net3_n_weights, 7160)

    def test_net4(self):
        self.assertEqual(self.net4_shape[1:3], [self.num_steps, self.hidden_size * 2])
        self.assertEqual(len(self.net4_layers), 2)
        self.assertEqual(len(self.net4_weights), 9)
        self.assertEqual(self.net4_n_weights, 13720)

    def test_net5(self):
        self.assertEqual(self.net5_shape[-1], self.vocab_size)
        self.assertEqual(self.net5_rnn_shape[-1], self.embedding_size)
        self.assertEqual(len(self.net5_layers), 3)
        self.assertEqual(len(self.net5_weights), 5)
        self.assertEqual(self.net5_n_weights, 4510)

    def test_net6(self):
        self.assertEqual(self.net6_shape[-1], self.embedding_size)
        self.assertEqual(self.net6_rnn_shape[-1], self.embedding_size)

    def test_net7(self):
        self.assertEqual(self.net7_shape[-1], self.embedding_size)

    def test_net8(self):
        self.assertEqual(self.net8_shape[-1], self.embedding_size)

    def test_net9(self):
        self.assertEqual(self.net9_shape[-1], self.vocab_size)
        self.assertEqual(self.net9_rnn_shape[-1], self.embedding_size * 2)
        self.assertEqual(len(self.net9_layers), 3)
        self.assertEqual(len(self.net9_weights), 7)
        self.assertEqual(self.net9_n_weights, 8390)

    def test_net10(self):
        self.assertEqual(self.net10_shape[-1], self.vocab_size)
        self.assertEqual(self.net10_rnn_shape[-1], self.embedding_size * 2)
        self.assertEqual(len(self.net10_layers), 3)
        self.assertEqual(len(self.net10_weights), 11)
        self.assertEqual(self.net10_n_weights, 18150)

    def test_net11(self):
        self.assertEqual(self.net11_shape[-1], 10000)
        self.assertEqual(len(self.net11_layers), 5)
        self.assertEqual(len(self.net11_weights), 11)
        self.assertEqual(self.net11_n_weights, 5293200)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
