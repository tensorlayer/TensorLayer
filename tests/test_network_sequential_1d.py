#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

try:
    from tests.unittests_helper import CustomTestCase
except ImportError:
    from unittests_helper import CustomTestCase


class Network_Sequential_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            cls.model = tl.networks.Sequential(name="My_Sequential_1D_Network")

            cls.model.add(tl.layers.ExpandDimsLayer(axis=1, name="expand_layer_1"))
            cls.model.add(tl.layers.FlattenLayer(name="flatten_layer_1"))

            cls.model.add(tl.layers.ExpandDimsLayer(axis=2, name="expand_layer_2"))
            cls.model.add(tl.layers.TileLayer(multiples=[1, 1, 3], name="tile_layer_2"))
            cls.model.add(tl.layers.TransposeLayer(perm=[0, 2, 1], name='transpose_layer_2'))
            cls.model.add(tl.layers.FlattenLayer(name="flatten_layer_2"))

            cls.model.add(tl.layers.DenseLayer(n_units=10, act=tf.nn.relu, name="seq_layer_1"))

            cls.model.add(tl.layers.DenseLayer(n_units=20, act=None, name="seq_layer_2"))
            cls.model.add(tl.layers.PReluLayer(channel_shared=True, name="prelu_layer_2"))

            cls.model.add(tl.layers.DenseLayer(n_units=30, act=None, name="seq_layer_3"))
            cls.model.add(tl.layers.PReluLayer(channel_shared=False, name="prelu_layer_3"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=None, name="seq_layer_4"))
            cls.model.add(tl.layers.PRelu6Layer(channel_shared=True, name="prelu6_layer_4"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=None, name="seq_layer_5"))
            cls.model.add(tl.layers.PRelu6Layer(channel_shared=False, name="prelu6_layer_5"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=None, name="seq_layer_6"))
            cls.model.add(tl.layers.PTRelu6Layer(channel_shared=True, name="ptrelu6_layer_6"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=None, name="seq_layer_7"))
            cls.model.add(tl.layers.PTRelu6Layer(channel_shared=False, name="ptrelu6_layer_7"))

            cls.model.add(tl.layers.DenseLayer(n_units=40, act=tf.nn.relu, name="seq_layer_8"))
            cls.model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=True, name="dropout_layer_8"))

            cls.model.add(tl.layers.DenseLayer(n_units=50, act=tf.nn.relu, name="seq_layer_9"))
            cls.model.add(tl.layers.DropoutLayer(keep=0.5, is_fix=False, name="dropout_layer_9"))

            cls.model.add(
                tl.layers.SlimNetsLayer(
                    slim_layer=slim.fully_connected,
                    slim_args={
                        'num_outputs': 50,
                        'activation_fn': None
                    },
                    act=tf.nn.relu,
                    name="seq_layer_10"
                )
            )

            cls.model.add(
                tl.layers.KerasLayer(
                    keras_layer=keras.layers.Dense, keras_args={'units': 256}, act=tf.nn.relu, name="seq_layer_11"
                )
            )

            cls.model.add(tl.layers.LambdaLayer(fn=lambda x: 2 * x, name='lambda_layer_11'))
            cls.model.add(tl.layers.GaussianNoiseLayer(mean=0.0, stddev=1.0, name='noise_layer_11'))
            cls.model.add(tl.layers.BatchNormLayer(decay=0.9, epsilon=1e-5, act=None, name='batchnorm_layer_11'))
            cls.model.add(
                tl.layers.LayerNormLayer(
                    center=True,
                    scale=True,
                    begin_norm_axis=1,
                    begin_params_axis=-1,
                    act=None,
                    name='layernorm_layer_11'
                )
            )

            cls.model.add(tl.layers.ExpandDimsLayer(axis=2, name="expand_layer_12"))
            cls.model.add(tl.layers.PadLayer(padding=[[0, 0], [4, 4], [0, 0]], name='pad_layer_12'))
            cls.model.add(tl.layers.ZeroPad1d(padding=1, name='zeropad1d_layer_12-1'))
            cls.model.add(tl.layers.ZeroPad1d(padding=(2, 3), name='zeropad1d_layer_12-2'))
            cls.model.add(tl.layers.ReshapeLayer(shape=(-1, 271), name='reshape_layer_12'))
            cls.model.add(tl.layers.ScaleLayer(init_scale=2., name='scale_layer_12'))

            cls.model.add(tl.layers.ReshapeLayer(shape=(-1, 271, 1), name='reshape_layer_13'))
            cls.model.add(
                tl.layers.Conv1dLayer(
                    shape=(5, 1, 12), stride=1, padding='SAME', act=tf.nn.relu, name='conv1d_layer_13'
                )
            )

            cls.model.add(
                tl.layers.Conv1dLayer(
                    shape=(5, 12, 24), stride=1, padding='SAME', b_init=None, act=tf.nn.relu, name='conv1d_layer_14'
                )
            )

            cls.model.add(
                tl.layers.Conv1d(
                    n_filter=12, filter_size=5, stride=1, padding='SAME', act=tf.nn.relu, name='conv1d_layer_15'
                )
            )

            cls.model.add(
                tl.layers.Conv1d(
                    n_filter=8, filter_size=5, stride=1, padding='SAME', b_init=None, act=None, name='conv1d_layer_16'
                )
            )

            cls.model.add(tl.layers.SubpixelConv1d(scale=2, name='subpixelconv1d_layer_17'))

            cls.model.add(
                tl.layers.SeparableConv1d(
                    n_filter=8,
                    filter_size=5,
                    strides=1,
                    padding='SAME',
                    act=tf.nn.relu,
                    name='separableconv1d_layer_18'
                )
            )

            cls.model.add(
                tl.layers.SeparableConv1d(
                    n_filter=4,
                    filter_size=5,
                    strides=1,
                    padding='SAME',
                    act=tf.nn.relu,
                    b_init=None,
                    name='separableconv1d_layer_19'
                )
            )

            plh = tf.placeholder(tf.float16, (100, 32))

            cls.train_model = cls.model.compile(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.compile(plh, reuse=True, is_train=False)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 1)

    def test_count_params(self):
        self.assertEqual(self.train_model.count_params(), 34193)

    def test_count_param_tensors(self):
        self.assertEqual(len(self.train_model.all_params), 48)

    def test_count_layers(self):
        self.assertEqual(len(self.train_model.all_layers), 44)

    def test_network_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.model.all_layers_dict.keys():
                if self.model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.model[layer_name].outputs.dtype, tf.float16)
                    )

    def test_network_shapes(self):

        self.assertEqual(self.model["input_layer"].outputs.shape, (100, 32))

        self.assertEqual(self.model["expand_layer_1"].outputs.shape, (100, 1, 32))
        self.assertEqual(self.model["flatten_layer_1"].outputs.shape, (100, 32))

        self.assertEqual(self.model["expand_layer_2"].outputs.shape, (100, 32, 1))
        self.assertEqual(self.model["tile_layer_2"].outputs.shape, (100, 32, 3))
        self.assertEqual(self.model["transpose_layer_2"].outputs.shape, (100, 3, 32))
        self.assertEqual(self.model["flatten_layer_2"].outputs.shape, (100, 96))

        self.assertEqual(self.model["seq_layer_1"].outputs.shape, (100, 10))

        self.assertEqual(self.model["seq_layer_2"].outputs.shape, (100, 20))
        self.assertEqual(self.model["prelu_layer_2"].outputs.shape, (100, 20))

        self.assertEqual(self.model["seq_layer_3"].outputs.shape, (100, 30))
        self.assertEqual(self.model["prelu_layer_3"].outputs.shape, (100, 30))

        self.assertEqual(self.model["seq_layer_4"].outputs.shape, (100, 40))
        self.assertEqual(self.model["prelu6_layer_4"].outputs.shape, (100, 40))

        self.assertEqual(self.model["seq_layer_5"].outputs.shape, (100, 50))
        self.assertEqual(self.model["prelu6_layer_5"].outputs.shape, (100, 50))

        self.assertEqual(self.model["seq_layer_6"].outputs.shape, (100, 40))
        self.assertEqual(self.model["ptrelu6_layer_6"].outputs.shape, (100, 40))

        self.assertEqual(self.model["seq_layer_7"].outputs.shape, (100, 50))
        self.assertEqual(self.model["ptrelu6_layer_7"].outputs.shape, (100, 50))

        self.assertEqual(self.model["seq_layer_8"].outputs.shape, (100, 40))
        self.assertEqual(self.model["dropout_layer_8"].outputs.shape, (100, 40))

        self.assertEqual(self.model["seq_layer_9"].outputs.shape, (100, 50))
        self.assertEqual(self.model["dropout_layer_9"].outputs.shape, (100, 50))

        self.assertEqual(self.model["seq_layer_10"].outputs.shape, (100, 50))

        self.assertEqual(self.model["seq_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.model["lambda_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.model["noise_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.model["batchnorm_layer_11"].outputs.shape, (100, 256))

        self.assertEqual(self.model["expand_layer_12"].outputs.shape, (100, 256, 1))
        self.assertEqual(self.model["pad_layer_12"].outputs.shape, (100, 264, 1))
        self.assertEqual(self.model["zeropad1d_layer_12-1"].outputs.shape, (100, 266, 1))
        self.assertEqual(self.model["zeropad1d_layer_12-2"].outputs.shape, (100, 271, 1))
        self.assertEqual(self.model["reshape_layer_12"].outputs.shape, (100, 271))
        self.assertEqual(self.model["scale_layer_12"].outputs.shape, (100, 271))

        self.assertEqual(self.model["reshape_layer_13"].outputs.shape, (100, 271, 1))
        self.assertEqual(self.model["conv1d_layer_13"].outputs.shape, (100, 271, 12))

        self.assertEqual(self.model["conv1d_layer_14"].outputs.shape, (100, 271, 24))

        self.assertEqual(self.model["conv1d_layer_15"].outputs.shape, (100, 271, 12))

        self.assertEqual(self.model["conv1d_layer_16"].outputs.shape, (100, 271, 8))

        self.assertEqual(self.model["subpixelconv1d_layer_17"].outputs.shape, (100, 542, 4))

        self.assertEqual(self.model["separableconv1d_layer_18"].outputs.shape, (100, 542, 8))

        self.assertEqual(self.model["separableconv1d_layer_19"].outputs.shape, (100, 542, 4))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
