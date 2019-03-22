#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_Lambda_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_x = np.random.random([100, 1]).astype(np.float32)
        cls.data_y = cls.data_x**3 + np.random.random() * cls.data_x**2 + np.random.random() * cls.data_x

    @classmethod
    def tearDownClass(cls):
        pass

    def test_lambda_keras(self):
        layers = [
            tf.keras.layers.Dense(10, activation=tf.nn.relu),
            tf.keras.layers.Dense(5, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(1, activation=tf.identity)
        ]
        perceptron = tf.keras.Sequential(layers)
        # in order to get trainable_variables of keras
        _ = perceptron(np.random.random([100, 5]).astype(np.float32))

        class CustomizeModel(tl.models.Model):
            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.Lambda(perceptron, perceptron.trainable_variables)

            def forward(self, x):
                z = self.dense(x)
                z = self.lambdalayer(z)
                return z

        optimizer = tf.optimizers.Adam(learning_rate=0.1)

        model = CustomizeModel()
        print(model.lambdalayer)

        model.train()

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y = model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, model.weights)
            optimizer.apply_gradients(zip(gradients, model.weights))

            print("epoch %d, loss %f" % (epoch, loss))

    def test_lambda_func_with_args(self):
        def customize_func(x, foo):
            if foo == 0:
                return tf.nn.relu(x)
            elif foo == 1:
                return tf.nn.sigmoid(x)
            else:
                return tf.identity(x)

        class CustomizeModel(tl.models.Model):
            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.Lambda(customize_func, fn_weights=[], fn_args={'foo': 0})

            def forward(self, x, foo):
                z = self.dense(x)
                if foo == -1:
                    zf = self.lambdalayer(z)
                else:
                    zf = self.lambdalayer(z, foo=foo)
                return z, zf

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        out, out2 = model(self.data_x, foo=-1)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.relu(out).numpy()))
        out, out2 = model(self.data_x, foo=0)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.relu(out).numpy()))
        out, out2 = model(self.data_x, foo=1)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.sigmoid(out).numpy()))
        out, out2 = model(self.data_x, foo=2)
        self.assertTrue(np.array_equal(out2.numpy(), out.numpy()))

    def test_lambda_func_without_args(self):

        class CustomizeModel(tl.models.Model):
            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.Lambda(lambda x: 2*x, fn_weights=[])

            def forward(self, x):
                z = self.dense(x)
                zf = self.lambdalayer(z)
                return z, zf

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        out, out2 = model(self.data_x)
        self.assertTrue(np.array_equal(out2.numpy(), out.numpy()*2))



if __name__ == '__main__':

    unittest.main()
