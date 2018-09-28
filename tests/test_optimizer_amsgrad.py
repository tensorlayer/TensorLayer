#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Pooling_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        cls.y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

        # define the network
        cls.network = tl.layers.InputLayer(cls.x, name='input')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.8, name='drop1')
        cls.network = tl.layers.DenseLayer(cls.network, 800, tf.nn.relu, name='relu1')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.5, name='drop2')
        cls.network = tl.layers.DenseLayer(cls.network, 800, tf.nn.relu, name='relu2')
        cls.network = tl.layers.DropoutLayer(cls.network, keep=0.5, name='drop3')

        cls.network = tl.layers.DenseLayer(cls.network, n_units=10, name='output')

        # define cost function and metric.
        cls.y = cls.network.outputs
        cls.cost = tl.cost.cross_entropy(cls.y, cls.y_, name='cost')

        correct_prediction = tf.equal(tf.argmax(cls.y, 1), cls.y_)

        cls.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # define the optimizer
        train_params = cls.network.all_params
        optimizer = tl.optimizers.AMSGrad(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8)
        cls.train_op = optimizer.minimize(cls.cost, var_list=train_params)

    @classmethod
    def tearDownClass(cls):
        tf.reset_default_graph()

    def test_training(self):

        with self.assertNotRaises(Exception):

            X_train, y_train, X_val, y_val, _, _ = tl.files.load_mnist_dataset(shape=(-1, 784))

            with tf.Session() as sess:
                # initialize all variables in the session
                tl.layers.initialize_global_variables(sess)

                # print network information
                self.network.print_params()
                self.network.print_layers()

                # train the network
                tl.utils.fit(
                    sess, self.network, self.train_op, self.cost, X_train, y_train, self.x, self.y_, acc=self.acc,
                    batch_size=500, n_epoch=1, print_freq=1, X_val=X_val, y_val=y_val, eval_train=False
                )


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
