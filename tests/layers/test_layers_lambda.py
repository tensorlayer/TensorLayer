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

        for epoch in range(10):
            with tf.GradientTape() as tape:
                pred_y = model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            print("epoch %d, loss %f" % (epoch, loss))

    def test_lambda_func_with_args(self):

        def customize_func(x, foo=42):
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

            def forward(self, x, bar):
                z = self.dense(x)
                if bar == -1:
                    zf = self.lambdalayer(z)
                else:
                    zf = self.lambdalayer(z, foo=bar)
                return z, zf

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        out, out2 = model(self.data_x, bar=-1)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.relu(out).numpy()))
        out, out2 = model(self.data_x, bar=0)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.relu(out).numpy()))
        out, out2 = model(self.data_x, bar=1)
        self.assertTrue(np.array_equal(out2.numpy(), tf.nn.sigmoid(out).numpy()))
        out, out2 = model(self.data_x, bar=2)
        self.assertTrue(np.array_equal(out2.numpy(), out.numpy()))

    def test_lambda_func_with_weight(self):

        a = tf.Variable(1.0)

        def customize_fn(x):
            return x + a

        class CustomizeModel(tl.models.Model):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.Lambda(customize_fn, fn_weights=[a])

            def forward(self, x):
                z = self.dense(x)
                z = self.lambdalayer(z)
                return z

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        out = model(self.data_x)
        print(out.shape)

    def test_lambda_func_without_args(self):

        class CustomizeModel(tl.models.Model):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.Lambda(lambda x: 2 * x)

            def forward(self, x):
                z = self.dense(x)
                zf = self.lambdalayer(z)
                return z, zf

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        out, out2 = model(self.data_x)
        self.assertTrue(np.array_equal(out2.numpy(), out.numpy() * 2))

    def test_elementwiselambda_func_with_args(self):

        def customize_func(noise, mean, std, foo=42):
            return mean + noise * tf.exp(std * 0.5) + foo

        class CustomizeModel(tl.models.Model):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense1 = tl.layers.Dense(in_channels=1, n_units=5)
                self.dense2 = tl.layers.Dense(in_channels=1, n_units=5)
                self.dense3 = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.ElementwiseLambda(customize_func, fn_args={'foo': 1024})

            def forward(self, x, bar=None):
                noise = self.dense1(x)
                mean = self.dense2(x)
                std = self.dense3(x)
                if bar is None:
                    out = self.lambdalayer([noise, mean, std])
                else:
                    out = self.lambdalayer([noise, mean, std], foo=bar)
                return noise, mean, std, out

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        noise, mean, std, out = model(self.data_x)
        self.assertTrue(np.allclose(out.numpy(), customize_func(noise, mean, std, foo=1024).numpy()))
        noise, mean, std, out = model(self.data_x, bar=2048)
        self.assertTrue(np.allclose(out.numpy(), customize_func(noise, mean, std, foo=2048).numpy()))

    def test_elementwiselambda_func_without_args(self):

        def customize_func(noise, mean, std):
            return mean + noise * tf.exp(std * 0.5)

        class CustomizeModel(tl.models.Model):

            def __init__(self):
                super(CustomizeModel, self).__init__()
                self.dense1 = tl.layers.Dense(in_channels=1, n_units=5)
                self.dense2 = tl.layers.Dense(in_channels=1, n_units=5)
                self.dense3 = tl.layers.Dense(in_channels=1, n_units=5)
                self.lambdalayer = tl.layers.ElementwiseLambda(customize_func, fn_weights=[])

            def forward(self, x):
                noise = self.dense1(x)
                mean = self.dense2(x)
                std = self.dense3(x)
                out = self.lambdalayer([noise, mean, std])
                return noise, mean, std, out

        model = CustomizeModel()
        print(model.lambdalayer)
        model.train()

        noise, mean, std, out = model(self.data_x)
        self.assertTrue(np.array_equal(out.numpy(), customize_func(noise, mean, std).numpy()))


if __name__ == '__main__':

    unittest.main()
