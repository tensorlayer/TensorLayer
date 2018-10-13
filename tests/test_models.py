#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class VGG_Model_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.Graph().as_default():
            # - Classify ImageNet classes with VGG16, see `tutorial_models_vgg16.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_vgg16.py>__`
            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            # get the whole model
            vgg1 = tl.models.VGG16(x)
            # restore pre-trained VGG parameters
            # sess = tf.InteractiveSession()
            # vgg.restore_weights(sess)
            # use for inferencing
            # probs = tf.nn.softmax(vgg1.outputs)

            cls.vgg1_layers = vgg1.all_layers
            cls.vgg1_weights = vgg1.all_weights

        with tf.Graph().as_default():
            # - Extract features with VGG16 and Train a classifier with 100 classes
            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            # get VGG without the last layer
            vgg2 = tl.models.VGG16(x, end_with='fc2_relu')

            cls.vgg2_layers = vgg2.all_layers
            cls.vgg2_weights = vgg2.all_weights

            print("TYPE:", type(vgg2))

            # add one more layer
            _ = tl.layers.Dense(n_units=100, name='out')(vgg2)
            # initialize all parameters
            # sess = tf.InteractiveSession()
            # tl.layers.initialize_global_variables(sess)
            # restore pre-trained VGG parameters
            # vgg.restore_weights(sess)
            # train your own classifier (only update the last layer)

            cls.vgg2_train_weights = tl.layers.get_variables_with_name('out')

        with tf.Graph().as_default() as graph:
            # - Reuse model
            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            # get VGG without the last layer
            vgg3 = tl.models.VGG16(x, end_with='fc2_relu')
            # reuse the parameters of vgg1 with different input
            # restore pre-trained VGG parameters (as they share parameters, we don’t need to restore vgg2)
            # sess = tf.InteractiveSession()
            # vgg1.restore_weights(sess)

            cls.vgg3_layers = vgg3.all_layers
            cls.vgg3_weights = vgg3.all_weights
            cls.vgg3_graph = graph

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_vgg1_layers(self):
        self.assertEqual(len(self.vgg1_layers), 23)

    def test_vgg2_layers(self):
        self.assertEqual(len(self.vgg2_layers), 22)

    def test_vgg3_layers(self):
        self.assertEqual(len(self.vgg3_layers), 22)

    def test_vgg1_weights(self):
        self.assertEqual(len(self.vgg1_weights), 32)

    def test_vgg2_weights(self):
        self.assertEqual(len(self.vgg2_weights), 30)

    def test_vgg3_weights(self):
        self.assertEqual(len(self.vgg3_weights), 30)

    def test_vgg2_train_weights(self):
        self.assertEqual(len(self.vgg2_train_weights), 2)

    def test_reuse_vgg(self):

        with self.assertNotRaises(Exception):
            with self.vgg3_graph.as_default():
                x = tf.placeholder(tf.float32, [None, 224, 224, 3])
                _ = tl.models.VGG16(x, end_with='fc2_relu', reuse=True)


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
