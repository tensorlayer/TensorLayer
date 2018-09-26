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
            cls.model.add(tl.layers.EmbeddingInputlayer(vocabulary_size=100, embedding_size=50, name='embedding'))
            cls.model.add(tl.layers.RNNLayer(n_hidden=100, n_steps=n_step, return_last=False, name='rnn'))
            cls.model.add(tl.layers.BiRNNLayer(n_hidden=50, n_steps=n_step, return_last=False, name='birnn'))
            cls.model.add(tl.layers.ReshapeLayer(shape=[-1, n_step, 5, 5, 4], name='reshape'))
            cls.model.add(
                tl.layers.ConvLSTMLayer(
                    # cell_fn=tl.layers.BasicConvLSTMCell(shape=(5, 5), filter_size=(3, 3), num_features=10),
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

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.test_model, tl.models.CompiledNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
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

    def test_count_weights_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 1 + 2 + 4 + 2)
        self.assertEqual(len(self.test_model.get_all_weights()), 1 + 2 + 4 + 2)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 1 + 2 + 4 + 2)

    def test_count_layers(self):
        print(self.train_model.count_layers())
        self.assertEqual(self.train_model.count_layers(), 6)
        self.assertEqual(self.test_model.count_layers(), 6)
        self.assertEqual(self.model.count_layers(), 6)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            self.assertEqual(self.train_model["embedding"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["embedding"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["rnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["rnn"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["birnn"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["birnn"].outputs.dtype, tf.float32)

            self.assertEqual(self.train_model["convlstm"].outputs.dtype, tf.float32)
            self.assertEqual(self.test_model["convlstm"].outputs.dtype, tf.float32)
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

        self.assertEqual(self.train_model["embedding"].outputs.shape, (100, 8, 50))
        self.assertEqual(self.test_model["embedding"].outputs.shape, (100, 8, 50))
        self.assertEqual(self.train_model["rnn"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.test_model["rnn"].outputs.shape, (100, 8, 100))
        self.assertEqual(self.train_model["birnn"].outputs.shape, (100, 8, 50 * 2))  # birnn x 2
        self.assertEqual(self.test_model["birnn"].outputs.shape, (100, 8, 50 * 2))
        self.assertEqual(self.train_model["convlstm"].outputs.shape, (100, 8, 5, 5, 5))
        self.assertEqual(self.test_model["convlstm"].outputs.shape, (100, 8, 5, 5, 5))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
