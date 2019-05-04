#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tests.utils import CustomTestCase


class Layer_Embed_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_onehot(self):
        input = tl.layers.Input([32], dtype=tf.int32)
        onehot = tl.layers.OneHot(depth=8, on_value=1, off_value=0, axis=-1)
        print(onehot)
        tensor = tl.layers.OneHot(depth=8)(input)
        self.assertEqual(tensor.get_shape().as_list(), [32, 8])
        model = tl.models.Model(inputs=input, outputs=tensor)

    def test_embed(self):
        input = tl.layers.Input([8, 100], dtype=tf.int32)
        embed = tl.layers.Embedding(vocabulary_size=1000, embedding_size=50, name='embed')
        print(embed)
        tensor = embed(input)
        self.assertEqual(tensor.get_shape().as_list(), [8, 100, 50])
        model = tl.models.Model(inputs=input, outputs=tensor)

    def test_avg_embed(self):
        batch_size = 8
        length = 5
        input = tl.layers.Input([batch_size, length], dtype=tf.int32)
        avgembed = tl.layers.AverageEmbedding(vocabulary_size=1000, embedding_size=50, name='avg')
        print(avgembed)
        tensor = avgembed(input)
        # print(tensor)
        self.assertEqual(tensor.get_shape().as_list(), [batch_size, 50])
        model = tl.models.Model(inputs=input, outputs=tensor)

    def test_word2vec_nce(self):
        batch_size = 8
        embedding_size = 50
        inputs = tl.layers.Input([batch_size], dtype=tf.int32)
        labels = tl.layers.Input([batch_size, 1], dtype=tf.int32)
        emb_net = tl.layers.Word2vecEmbedding(
            vocabulary_size=10000,
            embedding_size=embedding_size,
            num_sampled=100,
            activate_nce_loss=True,  # the nce loss is activated
            nce_loss_args={},
            E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
            nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))),
            nce_b_init=tl.initializers.constant(value=0.0),
        )
        print(emb_net)
        try:
            embed_tensor, embed_nce_loss = emb_net(inputs)
        except ValueError as e:
            print(e)
        try:
            embed_tensor = emb_net(inputs, use_nce_loss=False)
            print("Not use NCE without labels")
        except Exception as e:
            print(e)
        embed_tensor = emb_net([inputs, labels], use_nce_loss=False)
        embed_tensor, embed_nce_loss = emb_net([inputs, labels], use_nce_loss=True)
        embed_tensor, embed_nce_loss = emb_net([inputs, labels])
        self.assertEqual(embed_tensor.get_shape().as_list(), [batch_size, embedding_size])

        outputs = tl.layers.Dense(n_units=10)(embed_tensor)
        model = tl.models.Model(inputs=[inputs, labels], outputs=[outputs, embed_nce_loss])
        out, nce = model(
            [np.random.randint(0, 1, size=[batch_size]),
             np.random.randint(0, 1, size=[batch_size, 1])], is_train=True
        )
        self.assertEqual(out.get_shape().as_list(), [batch_size, 10])
        print(nce)

    def test_word2vec_no_nce(self):
        batch_size = 8
        embedding_size = 50
        inputs = tl.layers.Input([batch_size], dtype=tf.int32)
        emb_net = tl.layers.Word2vecEmbedding(
            vocabulary_size=10000,
            embedding_size=embedding_size,
            num_sampled=100,
            activate_nce_loss=False,  # the nce loss is activated
            nce_loss_args={},
            E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
            nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))),
            nce_b_init=tl.initializers.constant(value=0.0),
        )
        print(emb_net)
        embed_tensor = emb_net(inputs)
        embed_tensor = emb_net(inputs, use_nce_loss=False)
        try:
            embed_tensor = emb_net(inputs, use_nce_loss=True)
        except AttributeError as e:
            print(e)
        self.assertEqual(embed_tensor.get_shape().as_list(), [batch_size, embedding_size])
        model = tl.models.Model(inputs=inputs, outputs=embed_tensor)


if __name__ == '__main__':

    unittest.main()
