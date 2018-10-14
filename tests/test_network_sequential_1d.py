#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

import tensorflow.contrib.slim as slim
import tensorflow.keras as keras

from tests.utils import CustomTestCase


class Network_Sequential_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):
            cls.model = tl.networks.Sequential(name="My_Sequential_1D_Network")

            cls.model.add(tl.layers.ExpandDims(axis=1, name="expand_layer_1"))
            cls.model.add(tl.layers.Flatten(name="flatten_layer_1"))

            cls.model.add(tl.layers.ExpandDims(axis=2, name="expand_layer_2"))
            cls.model.add(tl.layers.Tile(multiples=[1, 1, 3], name="tile_layer_2"))
            cls.model.add(tl.layers.Transpose(perm=[0, 2, 1], name='transpose_layer_2'))
            cls.model.add(tl.layers.Flatten(name="flatten_layer_2"))

            cls.model.add(tl.layers.Dense(n_units=10, act=tf.nn.relu, name="seq_layer_1"))

            cls.model.add(tl.layers.Dense(n_units=20, act=None, name="seq_layer_2"))
            cls.model.add(tl.layers.PRelu(channel_shared=True, name="prelu_layer_2"))

            cls.model.add(tl.layers.Dense(n_units=30, act=None, name="seq_layer_3"))
            cls.model.add(tl.layers.PRelu(channel_shared=False, name="prelu_layer_3"))

            cls.model.add(tl.layers.Dense(n_units=40, act=None, name="seq_layer_4"))
            cls.model.add(tl.layers.PRelu6(channel_shared=True, name="prelu6_layer_4"))

            cls.model.add(tl.layers.Dense(n_units=50, act=None, name="seq_layer_5"))
            cls.model.add(tl.layers.PRelu6(channel_shared=False, name="prelu6_layer_5"))

            cls.model.add(tl.layers.Dense(n_units=40, act=None, name="seq_layer_6"))
            cls.model.add(tl.layers.PTRelu6(channel_shared=True, name="ptrelu6_layer_6"))

            cls.model.add(tl.layers.Dense(n_units=50, act=None, name="seq_layer_7"))
            cls.model.add(tl.layers.PTRelu6(channel_shared=False, name="ptrelu6_layer_7"))

            cls.model.add(tl.layers.Dense(n_units=40, act=tf.nn.relu, name="seq_layer_8"))
            cls.model.add(tl.layers.Dropout(keep=0.5, is_fix=True, name="dropout_layer_8"))

            # with tf.variable_scope('test', reuse=True): # # TODO:
            #     cls.model.add(tl.layers.Dense(n_units=5, act=tf.nn.relu, name="seq_layer_9"))
            #     cls.model.add(tl.layers.Dropout(keep=0.5, is_fix=False, name="dropout_layer_9"))

            cls.model.add(tl.layers.Dense(n_units=50, act=tf.nn.relu, name="seq_layer_9"))
            cls.model.add(tl.layers.Dropout(keep=0.5, is_fix=False, name="dropout_layer_9"))

            cls.model.add(
                tl.layers.SlimNets(
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
                tl.layers.
                Keras(keras_layer=keras.layers.Dense, keras_args={'units': 256}, act=tf.nn.relu, name="seq_layer_11")
            )

            cls.model.add(tl.layers.Lambda(fn=lambda x: 2 * x, name='lambda_layer_11'))
            cls.model.add(tl.layers.GaussianNoise(mean=0.0, stddev=1.0, name='noise_layer_11'))
            cls.model.add(tl.layers.BatchNorm(decay=0.9, epsilon=1e-5, act=None, name='batchnorm_layer_11'))
            cls.model.add(
                tl.layers.LayerNorm(
                    center=True,
                    scale=True,
                    begin_norm_axis=1,
                    begin_params_axis=-1,
                    act=None,
                    name='layernorm_layer_11'
                )
            )

            cls.model.add(tl.layers.ExpandDims(axis=2, name="expand_layer_12"))
            cls.model.add(tl.layers.PadLayer(padding=[[0, 0], [4, 4], [0, 0]], name='pad_layer_12'))
            cls.model.add(tl.layers.ZeroPad1d(padding=1, name='zeropad1d_layer_12-1'))
            cls.model.add(tl.layers.ZeroPad1d(padding=(2, 3), name='zeropad1d_layer_12-2'))
            cls.model.add(tl.layers.Reshape(shape=(-1, 271), name='reshape_layer_12'))
            cls.model.add(tl.layers.Scale(init_scale=2., name='scale_layer_12'))

            cls.model.add(tl.layers.Reshape(shape=(-1, 271, 1), name='reshape_layer_13'))
            cls.model.add(
                tl.layers.
                Conv1dLayer(shape=(5, 1, 12), stride=1, padding='SAME', act=tf.nn.relu, name='conv1d_layer_13')
            )

            cls.model.add(
                tl.layers.Conv1dLayer(
                    shape=(5, 12, 24), stride=1, padding='SAME', b_init=None, act=tf.nn.relu, name='conv1d_layer_14'
                )
            )

            cls.model.add(
                tl.layers.
                Conv1d(n_filter=12, filter_size=5, stride=1, padding='SAME', act=tf.nn.relu, name='conv1d_layer_15')
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

            cls.model.add(tl.layers.Flatten(name="flatten_layer_20"))

            cls.model.add(tl.layers.BinaryDense(n_units=10, act=tf.nn.sigmoid, name='binary_dense_layer_21'))

            cls.model.add(tl.layers.DorefaDense(n_units=20, name='dorefa_dense_layer_22'))

            cls.model.add(tl.layers.DropconnectDense(keep=0.5, n_units=30, name='dropconnect_layer_23'))

            cls.model.add(tl.layers.QuantizedDenseWithBN(n_units=40, name='quant_dense_bn_layer_24'))

            cls.model.add(tl.layers.QuantizedDense(n_units=30, name='quant_dense_layer_25'))

            cls.model.add(tl.layers.TernaryDense(n_units=20, name='ternary_dense_layer_26'))
            cls.model.add(tl.layers.Reshape(shape=(-1, 10, 2), name='reshape_layer_26'))

            cls.model.add(tl.layers.MaxPool1d(filter_size=3, strides=2, padding='valid', name='maxpool_1d_layer_27'))

            cls.model.add(tl.layers.MeanPool1d(filter_size=3, strides=2, padding='same', name='meanpool_1d_layer_28'))

            cls.model.add(tl.layers.GlobalMaxPool1d(name='global_maxpool_1d_layer_29'))
            cls.model.add(tl.layers.ExpandDims(axis=1, name='expand_layer_29'))
            cls.model.add(tl.layers.Tile(multiples=[1, 100, 1], name='tile_layer_29'))

            cls.model.add(tl.layers.GlobalMeanPool1d(name='global_meanpool_1d_layer_30'))
            cls.model.add(tl.layers.ExpandDims(axis=1, name='expand_layer_30'))
            cls.model.add(tl.layers.Tile(multiples=[1, 100, 1], name='tile_layer_30'))

            cls.model.add(tl.layers.Sign(name='sign_layer_31'))

            plh = tf.placeholder(tf.float16, (100, 32))

            cls.train_model = cls.model.build(plh, reuse=False, is_train=True)
            cls.test_model = cls.model.build(plh, reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.Sequential)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 2)
        self.assertEqual(len(self.test_model.all_drop), 0)  # In test mode, No Dropout

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 59943)
        self.assertEqual(self.test_model.count_weights(), 59943)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 59943)

    def test_count_weights_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 63)
        self.assertEqual(len(self.test_model.get_all_weights()), 63)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 63)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 61)
        self.assertEqual(self.test_model.count_layers(), 61)
        self.assertEqual(self.model.count_layers(), 61)

    def test_layer_outputs_dtype(self):

        with self.assertNotRaises(RuntimeError):

            for layer_name in self.train_model.all_layers:

                if self.train_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Train Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.train_model[layer_name].outputs.dtype, tf.float16)
                    )

                if self.test_model[layer_name].outputs.dtype != tf.float16:
                    raise RuntimeError(
                        "[Test Model] - Layer `%s` has an output of type %s, expected %s" %
                        (layer_name, self.test_model[layer_name].outputs.dtype, tf.float16)
                    )

    def test_layer_local_weights(self):

        # for layer_name in self.train_model.all_layers:
        #    print("self.assertEqual(self.train_model['%s'].count_local_weights(), %d)" % (layer_name, self.train_model[layer_name].count_local_weights()))
        #    print("self.assertEqual(self.test_model['%s'].count_local_weights(), %d)" % (layer_name, self.test_model[layer_name].count_local_weights()))
        #    print()

        self.assertEqual(self.train_model['input_layer'].count_local_weights(), 0)
        self.assertEqual(self.test_model['input_layer'].count_local_weights(), 0)

        self.assertEqual(self.train_model['expand_layer_1'].count_local_weights(), 0)
        self.assertEqual(self.test_model['expand_layer_1'].count_local_weights(), 0)

        self.assertEqual(self.train_model['flatten_layer_1'].count_local_weights(), 0)
        self.assertEqual(self.test_model['flatten_layer_1'].count_local_weights(), 0)

        self.assertEqual(self.train_model['expand_layer_2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['expand_layer_2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['tile_layer_2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['tile_layer_2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['transpose_layer_2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['transpose_layer_2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['flatten_layer_2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['flatten_layer_2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['seq_layer_1'].count_local_weights(), 970)
        self.assertEqual(self.test_model['seq_layer_1'].count_local_weights(), 970)

        self.assertEqual(self.train_model['seq_layer_2'].count_local_weights(), 220)
        self.assertEqual(self.test_model['seq_layer_2'].count_local_weights(), 220)

        self.assertEqual(self.train_model['prelu_layer_2'].count_local_weights(), 1)
        self.assertEqual(self.test_model['prelu_layer_2'].count_local_weights(), 1)

        self.assertEqual(self.train_model['seq_layer_3'].count_local_weights(), 630)
        self.assertEqual(self.test_model['seq_layer_3'].count_local_weights(), 630)

        self.assertEqual(self.train_model['prelu_layer_3'].count_local_weights(), 30)
        self.assertEqual(self.test_model['prelu_layer_3'].count_local_weights(), 30)

        self.assertEqual(self.train_model['seq_layer_4'].count_local_weights(), 1240)
        self.assertEqual(self.test_model['seq_layer_4'].count_local_weights(), 1240)

        self.assertEqual(self.train_model['prelu6_layer_4'].count_local_weights(), 1)
        self.assertEqual(self.test_model['prelu6_layer_4'].count_local_weights(), 1)

        self.assertEqual(self.train_model['seq_layer_5'].count_local_weights(), 2050)
        self.assertEqual(self.test_model['seq_layer_5'].count_local_weights(), 2050)

        self.assertEqual(self.train_model['prelu6_layer_5'].count_local_weights(), 50)
        self.assertEqual(self.test_model['prelu6_layer_5'].count_local_weights(), 50)

        self.assertEqual(self.train_model['seq_layer_6'].count_local_weights(), 2040)
        self.assertEqual(self.test_model['seq_layer_6'].count_local_weights(), 2040)

        self.assertEqual(self.train_model['ptrelu6_layer_6'].count_local_weights(), 2)
        self.assertEqual(self.test_model['ptrelu6_layer_6'].count_local_weights(), 2)

        self.assertEqual(self.train_model['seq_layer_7'].count_local_weights(), 2050)
        self.assertEqual(self.test_model['seq_layer_7'].count_local_weights(), 2050)

        self.assertEqual(self.train_model['ptrelu6_layer_7'].count_local_weights(), 100)
        self.assertEqual(self.test_model['ptrelu6_layer_7'].count_local_weights(), 100)

        self.assertEqual(self.train_model['seq_layer_8'].count_local_weights(), 2040)
        self.assertEqual(self.test_model['seq_layer_8'].count_local_weights(), 2040)

        self.assertEqual(self.train_model['dropout_layer_8'].count_local_weights(), 0)
        self.assertEqual(self.test_model['dropout_layer_8'].count_local_weights(), 0)

        self.assertEqual(self.train_model['seq_layer_9'].count_local_weights(), 2050)
        self.assertEqual(self.test_model['seq_layer_9'].count_local_weights(), 2050)

        self.assertEqual(self.train_model['dropout_layer_9'].count_local_weights(), 0)
        self.assertEqual(self.test_model['dropout_layer_9'].count_local_weights(), 0)

        self.assertEqual(self.train_model['seq_layer_10'].count_local_weights(), 2550)
        self.assertEqual(self.test_model['seq_layer_10'].count_local_weights(), 2550)

        self.assertEqual(self.train_model['seq_layer_11'].count_local_weights(), 13056)
        self.assertEqual(self.test_model['seq_layer_11'].count_local_weights(), 13056)

        self.assertEqual(self.train_model['lambda_layer_11'].count_local_weights(), 0)
        self.assertEqual(self.test_model['lambda_layer_11'].count_local_weights(), 0)

        self.assertEqual(self.train_model['noise_layer_11'].count_local_weights(), 0)
        self.assertEqual(self.test_model['noise_layer_11'].count_local_weights(), 0)

        self.assertEqual(self.train_model['batchnorm_layer_11'].count_local_weights(), 1024)
        self.assertEqual(self.test_model['batchnorm_layer_11'].count_local_weights(), 1024)

        self.assertEqual(self.train_model['layernorm_layer_11'].count_local_weights(), 512)
        self.assertEqual(self.test_model['layernorm_layer_11'].count_local_weights(), 512)

        self.assertEqual(self.train_model['expand_layer_12'].count_local_weights(), 0)
        self.assertEqual(self.test_model['expand_layer_12'].count_local_weights(), 0)

        self.assertEqual(self.train_model['pad_layer_12'].count_local_weights(), 0)
        self.assertEqual(self.test_model['pad_layer_12'].count_local_weights(), 0)

        self.assertEqual(self.train_model['zeropad1d_layer_12-1'].count_local_weights(), 0)
        self.assertEqual(self.test_model['zeropad1d_layer_12-1'].count_local_weights(), 0)

        self.assertEqual(self.train_model['zeropad1d_layer_12-2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['zeropad1d_layer_12-2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['reshape_layer_12'].count_local_weights(), 0)
        self.assertEqual(self.test_model['reshape_layer_12'].count_local_weights(), 0)

        self.assertEqual(self.train_model['scale_layer_12'].count_local_weights(), 1)
        self.assertEqual(self.test_model['scale_layer_12'].count_local_weights(), 1)

        self.assertEqual(self.train_model['reshape_layer_13'].count_local_weights(), 0)
        self.assertEqual(self.test_model['reshape_layer_13'].count_local_weights(), 0)

        self.assertEqual(self.train_model['conv1d_layer_13'].count_local_weights(), 72)
        self.assertEqual(self.test_model['conv1d_layer_13'].count_local_weights(), 72)

        self.assertEqual(self.train_model['conv1d_layer_14'].count_local_weights(), 1440)
        self.assertEqual(self.test_model['conv1d_layer_14'].count_local_weights(), 1440)

        self.assertEqual(self.train_model['conv1d_layer_15'].count_local_weights(), 1452)
        self.assertEqual(self.test_model['conv1d_layer_15'].count_local_weights(), 1452)

        self.assertEqual(self.train_model['conv1d_layer_16'].count_local_weights(), 480)
        self.assertEqual(self.test_model['conv1d_layer_16'].count_local_weights(), 480)

        self.assertEqual(self.train_model['subpixelconv1d_layer_17'].count_local_weights(), 0)
        self.assertEqual(self.test_model['subpixelconv1d_layer_17'].count_local_weights(), 0)

        self.assertEqual(self.train_model['separableconv1d_layer_18'].count_local_weights(), 60)
        self.assertEqual(self.test_model['separableconv1d_layer_18'].count_local_weights(), 60)

        self.assertEqual(self.train_model['separableconv1d_layer_19'].count_local_weights(), 72)
        self.assertEqual(self.test_model['separableconv1d_layer_19'].count_local_weights(), 72)

        self.assertEqual(self.train_model['flatten_layer_20'].count_local_weights(), 0)
        self.assertEqual(self.test_model['flatten_layer_20'].count_local_weights(), 0)

        self.assertEqual(self.train_model['binary_dense_layer_21'].count_local_weights(), 21690)
        self.assertEqual(self.test_model['binary_dense_layer_21'].count_local_weights(), 21690)

        self.assertEqual(self.train_model['dorefa_dense_layer_22'].count_local_weights(), 220)
        self.assertEqual(self.test_model['dorefa_dense_layer_22'].count_local_weights(), 220)

        self.assertEqual(self.train_model['dropconnect_layer_23'].count_local_weights(), 630)
        self.assertEqual(self.test_model['dropconnect_layer_23'].count_local_weights(), 630)

        self.assertEqual(self.train_model['quant_dense_bn_layer_24'].count_local_weights(), 1360)
        self.assertEqual(self.test_model['quant_dense_bn_layer_24'].count_local_weights(), 1360)

        self.assertEqual(self.train_model['quant_dense_layer_25'].count_local_weights(), 1230)
        self.assertEqual(self.test_model['quant_dense_layer_25'].count_local_weights(), 1230)

        self.assertEqual(self.train_model['ternary_dense_layer_26'].count_local_weights(), 620)
        self.assertEqual(self.test_model['ternary_dense_layer_26'].count_local_weights(), 620)

        self.assertEqual(self.train_model['reshape_layer_26'].count_local_weights(), 0)
        self.assertEqual(self.test_model['reshape_layer_26'].count_local_weights(), 0)

        self.assertEqual(self.train_model['maxpool_1d_layer_27'].count_local_weights(), 0)
        self.assertEqual(self.test_model['maxpool_1d_layer_27'].count_local_weights(), 0)

        self.assertEqual(self.train_model['meanpool_1d_layer_28'].count_local_weights(), 0)
        self.assertEqual(self.test_model['meanpool_1d_layer_28'].count_local_weights(), 0)

        self.assertEqual(self.train_model['global_maxpool_1d_layer_29'].count_local_weights(), 0)
        self.assertEqual(self.test_model['global_maxpool_1d_layer_29'].count_local_weights(), 0)

        self.assertEqual(self.train_model['expand_layer_29'].count_local_weights(), 0)
        self.assertEqual(self.test_model['expand_layer_29'].count_local_weights(), 0)

        self.assertEqual(self.train_model['tile_layer_29'].count_local_weights(), 0)
        self.assertEqual(self.test_model['tile_layer_29'].count_local_weights(), 0)

        self.assertEqual(self.train_model['global_meanpool_1d_layer_30'].count_local_weights(), 0)
        self.assertEqual(self.test_model['global_meanpool_1d_layer_30'].count_local_weights(), 0)

        self.assertEqual(self.train_model['expand_layer_30'].count_local_weights(), 0)
        self.assertEqual(self.test_model['expand_layer_30'].count_local_weights(), 0)

        self.assertEqual(self.train_model['tile_layer_30'].count_local_weights(), 0)
        self.assertEqual(self.test_model['tile_layer_30'].count_local_weights(), 0)

        self.assertEqual(self.train_model['sign_layer_31'].count_local_weights(), 0)
        self.assertEqual(self.test_model['sign_layer_31'].count_local_weights(), 0)

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_layer"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["input_layer"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["expand_layer_1"].outputs.shape, (100, 1, 32))
        self.assertEqual(self.test_model["expand_layer_1"].outputs.shape, (100, 1, 32))

        self.assertEqual(self.train_model["flatten_layer_1"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["flatten_layer_1"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["expand_layer_2"].outputs.shape, (100, 32, 1))
        self.assertEqual(self.test_model["expand_layer_2"].outputs.shape, (100, 32, 1))

        self.assertEqual(self.train_model["tile_layer_2"].outputs.shape, (100, 32, 3))
        self.assertEqual(self.test_model["tile_layer_2"].outputs.shape, (100, 32, 3))

        self.assertEqual(self.train_model["transpose_layer_2"].outputs.shape, (100, 3, 32))
        self.assertEqual(self.test_model["transpose_layer_2"].outputs.shape, (100, 3, 32))

        self.assertEqual(self.train_model["flatten_layer_2"].outputs.shape, (100, 96))
        self.assertEqual(self.test_model["flatten_layer_2"].outputs.shape, (100, 96))

        self.assertEqual(self.train_model["seq_layer_1"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["seq_layer_1"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["seq_layer_2"].outputs.shape, (100, 20))
        self.assertEqual(self.test_model["seq_layer_2"].outputs.shape, (100, 20))

        self.assertEqual(self.train_model["prelu_layer_2"].outputs.shape, (100, 20))
        self.assertEqual(self.test_model["prelu_layer_2"].outputs.shape, (100, 20))

        self.assertEqual(self.train_model["seq_layer_3"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["seq_layer_3"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["prelu_layer_3"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["prelu_layer_3"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["seq_layer_4"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["seq_layer_4"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["prelu6_layer_4"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["prelu6_layer_4"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["seq_layer_5"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["seq_layer_5"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["prelu6_layer_5"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["prelu6_layer_5"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["seq_layer_6"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["seq_layer_6"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["ptrelu6_layer_6"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["ptrelu6_layer_6"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["seq_layer_7"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["seq_layer_7"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["ptrelu6_layer_7"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["ptrelu6_layer_7"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["seq_layer_8"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["seq_layer_8"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["dropout_layer_8"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["dropout_layer_8"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["seq_layer_9"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["seq_layer_9"].outputs.shape, (100, 50))

        # self.assertEqual(self.train_model["test/seq_layer_9"].outputs.shape, (100, 50))
        # self.assertEqual(self.test_model["test/seq_layer_9"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["dropout_layer_9"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["dropout_layer_9"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["seq_layer_10"].outputs.shape, (100, 50))
        self.assertEqual(self.test_model["seq_layer_10"].outputs.shape, (100, 50))

        self.assertEqual(self.train_model["seq_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.test_model["seq_layer_11"].outputs.shape, (100, 256))

        self.assertEqual(self.train_model["lambda_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.test_model["lambda_layer_11"].outputs.shape, (100, 256))

        self.assertEqual(self.train_model["noise_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.test_model["noise_layer_11"].outputs.shape, (100, 256))

        self.assertEqual(self.train_model["batchnorm_layer_11"].outputs.shape, (100, 256))
        self.assertEqual(self.test_model["batchnorm_layer_11"].outputs.shape, (100, 256))

        self.assertEqual(self.train_model["expand_layer_12"].outputs.shape, (100, 256, 1))
        self.assertEqual(self.test_model["expand_layer_12"].outputs.shape, (100, 256, 1))

        self.assertEqual(self.train_model["pad_layer_12"].outputs.shape, (100, 264, 1))
        self.assertEqual(self.test_model["pad_layer_12"].outputs.shape, (100, 264, 1))

        self.assertEqual(self.train_model["zeropad1d_layer_12-1"].outputs.shape, (100, 266, 1))
        self.assertEqual(self.test_model["zeropad1d_layer_12-1"].outputs.shape, (100, 266, 1))

        self.assertEqual(self.train_model["zeropad1d_layer_12-2"].outputs.shape, (100, 271, 1))
        self.assertEqual(self.test_model["zeropad1d_layer_12-2"].outputs.shape, (100, 271, 1))

        self.assertEqual(self.train_model["reshape_layer_12"].outputs.shape, (100, 271))
        self.assertEqual(self.test_model["reshape_layer_12"].outputs.shape, (100, 271))

        self.assertEqual(self.train_model["scale_layer_12"].outputs.shape, (100, 271))
        self.assertEqual(self.test_model["scale_layer_12"].outputs.shape, (100, 271))

        self.assertEqual(self.train_model["reshape_layer_13"].outputs.shape, (100, 271, 1))
        self.assertEqual(self.test_model["reshape_layer_13"].outputs.shape, (100, 271, 1))

        self.assertEqual(self.train_model["conv1d_layer_13"].outputs.shape, (100, 271, 12))
        self.assertEqual(self.test_model["conv1d_layer_13"].outputs.shape, (100, 271, 12))

        self.assertEqual(self.train_model["conv1d_layer_14"].outputs.shape, (100, 271, 24))
        self.assertEqual(self.test_model["conv1d_layer_14"].outputs.shape, (100, 271, 24))

        self.assertEqual(self.train_model["conv1d_layer_15"].outputs.shape, (100, 271, 12))
        self.assertEqual(self.test_model["conv1d_layer_15"].outputs.shape, (100, 271, 12))

        self.assertEqual(self.train_model["conv1d_layer_16"].outputs.shape, (100, 271, 8))
        self.assertEqual(self.test_model["conv1d_layer_16"].outputs.shape, (100, 271, 8))

        self.assertEqual(self.train_model["subpixelconv1d_layer_17"].outputs.shape, (100, 542, 4))
        self.assertEqual(self.test_model["subpixelconv1d_layer_17"].outputs.shape, (100, 542, 4))

        self.assertEqual(self.train_model["separableconv1d_layer_18"].outputs.shape, (100, 542, 8))
        self.assertEqual(self.test_model["separableconv1d_layer_18"].outputs.shape, (100, 542, 8))

        self.assertEqual(self.train_model["separableconv1d_layer_19"].outputs.shape, (100, 542, 4))
        self.assertEqual(self.test_model["separableconv1d_layer_19"].outputs.shape, (100, 542, 4))

        self.assertEqual(self.train_model["flatten_layer_20"].outputs.shape, (100, 2168))
        self.assertEqual(self.test_model["flatten_layer_20"].outputs.shape, (100, 2168))

        self.assertEqual(self.train_model["binary_dense_layer_21"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["binary_dense_layer_21"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["dorefa_dense_layer_22"].outputs.shape, (100, 20))
        self.assertEqual(self.test_model["dorefa_dense_layer_22"].outputs.shape, (100, 20))

        self.assertEqual(self.train_model["dropconnect_layer_23"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["dropconnect_layer_23"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["quant_dense_bn_layer_24"].outputs.shape, (100, 40))
        self.assertEqual(self.test_model["quant_dense_bn_layer_24"].outputs.shape, (100, 40))

        self.assertEqual(self.train_model["quant_dense_layer_25"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["quant_dense_layer_25"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["ternary_dense_layer_26"].outputs.shape, (100, 20))
        self.assertEqual(self.test_model["ternary_dense_layer_26"].outputs.shape, (100, 20))

        self.assertEqual(self.train_model["reshape_layer_26"].outputs.shape, (100, 10, 2))
        self.assertEqual(self.test_model["reshape_layer_26"].outputs.shape, (100, 10, 2))

        self.assertEqual(self.train_model["maxpool_1d_layer_27"].outputs.shape, (100, 4, 2))
        self.assertEqual(self.test_model["maxpool_1d_layer_27"].outputs.shape, (100, 4, 2))

        self.assertEqual(self.train_model["meanpool_1d_layer_28"].outputs.shape, (100, 2, 2))
        self.assertEqual(self.test_model["meanpool_1d_layer_28"].outputs.shape, (100, 2, 2))

        self.assertEqual(self.train_model["global_maxpool_1d_layer_29"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["global_maxpool_1d_layer_29"].outputs.shape, (100, 2))

        self.assertEqual(self.train_model["expand_layer_29"].outputs.shape, (100, 1, 2))
        self.assertEqual(self.test_model["expand_layer_29"].outputs.shape, (100, 1, 2))

        self.assertEqual(self.train_model["tile_layer_29"].outputs.shape, (100, 100, 2))
        self.assertEqual(self.test_model["tile_layer_29"].outputs.shape, (100, 100, 2))

        self.assertEqual(self.train_model["global_meanpool_1d_layer_30"].outputs.shape, (100, 2))
        self.assertEqual(self.test_model["global_meanpool_1d_layer_30"].outputs.shape, (100, 2))

        self.assertEqual(self.train_model["expand_layer_30"].outputs.shape, (100, 1, 2))
        self.assertEqual(self.test_model["expand_layer_30"].outputs.shape, (100, 1, 2))

        self.assertEqual(self.train_model["tile_layer_30"].outputs.shape, (100, 100, 2))
        self.assertEqual(self.test_model["tile_layer_30"].outputs.shape, (100, 100, 2))

        self.assertEqual(self.train_model["sign_layer_31"].outputs.shape, (100, 100, 2))
        self.assertEqual(self.test_model["sign_layer_31"].outputs.shape, (100, 100, 2))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
