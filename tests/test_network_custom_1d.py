#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class CustomNetwork_1D_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        with tf.variable_scope("test_scope"):

            def my_custom_lambda_func(noise, mean, std):
                return mean + noise * tf.exp(std * 0.5)

            class MyCustomNetwork(tl.networks.CustomModel):

                def model(self):
                    input_layer = tl.layers.Input(name='input_layer')
                    noise_layer = tl.layers.Input(name="input_noise")

                    net_0 = tl.layers.Dense(n_units=24, act=tf.nn.relu, name='dense_layer_1')(input_layer)
                    net_1 = tl.layers.Dense(n_units=24, act=tf.nn.relu, name='dense_layer_2')(input_layer)

                    net = tl.layers.Elementwise(combine_fn=tf.minimum, name='elementwise_layer_3')([net_0, net_1])

                    mean = tl.layers.Dense(n_units=32, act=tf.nn.relu, name='dense_layer_4')(net)
                    std = tl.layers.Dense(n_units=32, act=tf.nn.relu, name='dense_layer_5')(net)

                    net = tl.layers.ElementwiseLambda(
                        fn=my_custom_lambda_func, name='elementwise_lambda_layer_6'
                    )([noise_layer, mean, std])

                    net_d1 = tl.layers.Dense(n_units=10, name='dense_layer_7_1')(net)
                    net_d2 = tl.layers.Dense(n_units=10, name='dense_layer_7_2')(net)
                    net_d3 = tl.layers.Dense(n_units=10, name='dense_layer_7_3')(net)

                    net_stack = tl.layers.Stack(axis=1, name='stack_layer_8')([net_d1, net_d2, net_d3])

                    net_unstack = tl.layers.UnStack(axis=1, name='unstack_layer_9')(net_stack)

                    net_unstacked_d1 = tl.layers.Lambda(
                        fn=lambda x: x[0].outputs, name="unstacked_layer_9_1"
                    )(net_unstack)

                    net_unstacked_d2 = tl.layers.Lambda(
                        fn=lambda x: x[1].outputs, name="unstacked_layer_9_2"
                    )(net_unstack)

                    net_unstacked_d3 = tl.layers.Lambda(
                        fn=lambda x: x[2].outputs, name="unstacked_layer_9_3"
                    )(net_unstack)

                    net = tl.layers.Concat(name='concat_layer_10'
                                          )([net_unstacked_d1, net_unstacked_d2, net_unstacked_d3])

                    return (input_layer, noise_layer), net

            cls.model = MyCustomNetwork(name="my_custom_network")

            plh = tf.placeholder(tf.float16, (100, 30))

            noise_tensor = tf.random_normal(shape=[tf.shape(plh)[0], 32], dtype=plh.dtype)

            cls.train_model = cls.model.build((plh, noise_tensor), reuse=False, is_train=True)
            cls.test_model = cls.model.build((plh, noise_tensor), reuse=True, is_train=False)

    def test_objects_dtype(self):
        self.assertIsInstance(self.train_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.test_model, tl.models.BuiltNetwork)
        self.assertIsInstance(self.model, tl.networks.CustomModel)

    def test_get_all_drop_plh(self):
        self.assertEqual(len(self.train_model.all_drop), 0)
        self.assertEqual(len(self.test_model.all_drop), 0)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.all_drop), 0)

    def test_count_weights(self):
        self.assertEqual(self.train_model.count_weights(), 4078)
        self.assertEqual(self.test_model.count_weights(), 4078)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(self.model.count_weights(), 3088)

    def test_count_weight_tensors(self):
        self.assertEqual(len(self.train_model.get_all_weights()), 14)
        self.assertEqual(len(self.test_model.get_all_weights()), 14)

        with self.assertRaises((AttributeError, AssertionError)):
            self.assertEqual(len(self.model.get_all_weights()), 14)

    def test_count_layers(self):
        self.assertEqual(self.train_model.count_layers(), 17)
        self.assertEqual(self.test_model.count_layers(), 17)
        self.assertEqual(self.model.count_layers(), 17)

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
        #     print("self.assertEqual(self.train_model['%s'].count_local_weights(), %d)" % (layer_name, self.train_model[layer_name].count_local_weights()))
        #     print("self.assertEqual(self.test_model['%s'].count_local_weights(), %d)" % (layer_name, self.test_model[layer_name].count_local_weights()))
        #     print()

        self.assertEqual(self.train_model['input_layer'].count_local_weights(), 0)
        self.assertEqual(self.test_model['input_layer'].count_local_weights(), 0)

        self.assertEqual(self.train_model['input_noise'].count_local_weights(), 0)
        self.assertEqual(self.test_model['input_noise'].count_local_weights(), 0)

        self.assertEqual(self.train_model['dense_layer_1'].count_local_weights(), 744)
        self.assertEqual(self.test_model['dense_layer_1'].count_local_weights(), 744)

        self.assertEqual(self.train_model['dense_layer_2'].count_local_weights(), 744)
        self.assertEqual(self.test_model['dense_layer_2'].count_local_weights(), 744)

        self.assertEqual(self.train_model['elementwise_layer_3'].count_local_weights(), 0)
        self.assertEqual(self.test_model['elementwise_layer_3'].count_local_weights(), 0)

        self.assertEqual(self.train_model['dense_layer_4'].count_local_weights(), 800)
        self.assertEqual(self.test_model['dense_layer_4'].count_local_weights(), 800)

        self.assertEqual(self.train_model['dense_layer_5'].count_local_weights(), 800)
        self.assertEqual(self.test_model['dense_layer_5'].count_local_weights(), 800)

        self.assertEqual(self.train_model['elementwise_lambda_layer_6'].count_local_weights(), 0)
        self.assertEqual(self.test_model['elementwise_lambda_layer_6'].count_local_weights(), 0)

        self.assertEqual(self.train_model['dense_layer_7_1'].count_local_weights(), 330)
        self.assertEqual(self.test_model['dense_layer_7_1'].count_local_weights(), 330)

        self.assertEqual(self.train_model['dense_layer_7_2'].count_local_weights(), 330)
        self.assertEqual(self.test_model['dense_layer_7_2'].count_local_weights(), 330)

        self.assertEqual(self.train_model['dense_layer_7_3'].count_local_weights(), 330)
        self.assertEqual(self.test_model['dense_layer_7_3'].count_local_weights(), 330)

        self.assertEqual(self.train_model['stack_layer_8'].count_local_weights(), 0)
        self.assertEqual(self.test_model['stack_layer_8'].count_local_weights(), 0)

        self.assertEqual(self.train_model['unstack_layer_9'].count_local_weights(), 0)
        self.assertEqual(self.test_model['unstack_layer_9'].count_local_weights(), 0)

        self.assertEqual(self.train_model['unstacked_layer_9_1'].count_local_weights(), 0)
        self.assertEqual(self.test_model['unstacked_layer_9_1'].count_local_weights(), 0)

        self.assertEqual(self.train_model['unstacked_layer_9_2'].count_local_weights(), 0)
        self.assertEqual(self.test_model['unstacked_layer_9_2'].count_local_weights(), 0)

        self.assertEqual(self.train_model['unstacked_layer_9_3'].count_local_weights(), 0)
        self.assertEqual(self.test_model['unstacked_layer_9_3'].count_local_weights(), 0)

        self.assertEqual(self.train_model['concat_layer_10'].count_local_weights(), 0)
        self.assertEqual(self.test_model['concat_layer_10'].count_local_weights(), 0)

    def test_network_shapes(self):

        self.assertEqual(self.train_model["input_layer"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["input_layer"].outputs.shape, (100, 30))

        self.assertEqual(self.train_model["input_noise"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["input_noise"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["dense_layer_1"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["dense_layer_1"].outputs.shape, (100, 24))

        self.assertEqual(self.train_model["dense_layer_2"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["dense_layer_2"].outputs.shape, (100, 24))

        self.assertEqual(self.train_model["elementwise_layer_3"].outputs.shape, (100, 24))
        self.assertEqual(self.test_model["elementwise_layer_3"].outputs.shape, (100, 24))

        self.assertEqual(self.train_model["dense_layer_4"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["dense_layer_4"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["dense_layer_5"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["dense_layer_5"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["elementwise_lambda_layer_6"].outputs.shape, (100, 32))
        self.assertEqual(self.test_model["elementwise_lambda_layer_6"].outputs.shape, (100, 32))

        self.assertEqual(self.train_model["dense_layer_7_1"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["dense_layer_7_1"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["dense_layer_7_2"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["dense_layer_7_2"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["dense_layer_7_3"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["dense_layer_7_3"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["stack_layer_8"].outputs.shape, (100, 3, 10))
        self.assertEqual(self.test_model["stack_layer_8"].outputs.shape, (100, 3, 10))

        self.assertEqual(self.train_model["unstack_layer_9"].outputs.shape, (3, 100, 10))
        self.assertEqual(self.test_model["unstack_layer_9"].outputs.shape, (3, 100, 10))

        self.assertEqual(self.train_model["unstacked_layer_9_1"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["unstacked_layer_9_1"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["unstacked_layer_9_2"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["unstacked_layer_9_2"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["unstacked_layer_9_3"].outputs.shape, (100, 10))
        self.assertEqual(self.test_model["unstacked_layer_9_3"].outputs.shape, (100, 10))

        self.assertEqual(self.train_model["concat_layer_10"].outputs.shape, (100, 30))
        self.assertEqual(self.test_model["concat_layer_10"].outputs.shape, (100, 30))


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
