#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope

slim = tf.contrib.slim
keras = tf.keras

import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Importer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.net_in = dict()

        # ============================= #
        #          LambdaLayer
        # ============================= #
        x = tf.placeholder(tf.float32, shape=[None, 784])
        cls.net_in["lambda"] = tl.layers.InputLayer(x, name='input')

        # ============================= #
        #          SlimNetsLayer
        # ============================= #
        x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        cls.net_in["slim"] = tl.layers.InputLayer(x, name='input_layer')

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_lambda_layer(self):

        def keras_block(x):
            x = keras.layers.Dropout(0.8)(x)
            x = keras.layers.Dense(100, activation='relu')(x)
            # x = keras.layers.Dropout(0.8)(x)
            # x = keras.layers.Dense(100, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            logits = keras.layers.Dense(10, activation='linear')(x)

            return logits

        with self.assertNotRaises(Exception):
            tl.layers.LambdaLayer(self.net_in["lambda"], fn=keras_block, name='keras')

    def test_slim_layer(self):

        with self.assertNotRaises(Exception):
            with slim.arg_scope(inception_v3_arg_scope()):
                # Alternatively, you should implement inception_v3 without TensorLayer as follow.
                # logits, end_points = inception_v3(X, num_classes=1001,
                #                                   is_training=False)
                tl.layers.SlimNetsLayer(
                    self.net_in["slim"],
                    slim_layer=inception_v3,
                    slim_args={
                        'num_classes': 1001,
                        'is_training': False,
                        #  'dropout_keep_prob' : 0.8,       # for training
                        #  'min_depth' : 16,
                        #  'depth_multiplier' : 1.0,
                        #  'prediction_fn' : slim.softmax,
                        #  'spatial_squeeze' : True,
                        #  'reuse' : None,
                        #  'scope' : 'InceptionV3'
                    },
                    name='InceptionV3'  # <-- the name should be the same with the ckpt model
                )


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
