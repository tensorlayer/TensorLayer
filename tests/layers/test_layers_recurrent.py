#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tests.utils import CustomTestCase


class Layer_RNN_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):

        cls.batch_size = 2

        cls.vocab_size = 20
        cls.embedding_size = 4

        cls.hidden_size = 8
        cls.num_steps = 6

        cls.data_x = np.random.random([cls.batch_size, cls.num_steps, cls.embedding_size]).astype(np.float32)
        cls.data_y = np.zeros([cls.batch_size, 1]).astype(np.float32)

        map1 = np.random.random([1, cls.num_steps])
        map2 = np.random.random([cls.embedding_size, 1])
        for i in range(cls.batch_size):
            cls.data_y[i] = np.matmul(map1, np.matmul(cls.data_x[i], map2))


    @classmethod
    def tearDownClass(cls):
        pass

    def test_basic_simplernn(self):

        inputs = tl.layers.Input([self.batch_size, self.num_steps, self.embedding_size])
        rnnlayer = tl.layers.RNN(
            cell=tf.keras.layers.SimpleRNNCell(units=self.hidden_size, dropout=0.1),
            return_last=True, return_seq_2d=False, return_state=True, name='simplernn'
        )
        rnn, rnn_state = rnnlayer(inputs)
        outputs = tl.layers.Dense(n_units=1)(rnn)
        rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0]], name='rnn_model')
        print(rnn_model)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        rnn_model.train()
        assert rnnlayer.is_train

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y, final_state = rnn_model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, rnn_model.weights)
            optimizer.apply_gradients(zip(gradients, rnn_model.weights))

            if (epoch + 1) % 10 == 0:
                print("epoch %d, loss %f" % (epoch, loss))

    def test_basic_simplernn_dynamic(self):

        class CustomisedModel(tl.models.Model):
            def __init__(self):
                super(CustomisedModel, self).__init__()
                self.rnnlayer = tl.layers.RNN(
                    cell=tf.keras.layers.SimpleRNNCell(units=8, dropout=0.1),
                    inputs_shape=[2, 6, 4],
                    return_last=False, return_seq_2d=False, return_state=False, name='simplernn'
                )
                self.dense = tl.layers.Dense(in_channels=8, n_units=1)

            def forward(self, x):
                z = self.rnnlayer(x)
                z = self.dense(z[:,-1,:])
                return z

        rnn_model = CustomisedModel()
        print(rnn_model)
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        rnn_model.train()

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y = rnn_model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, rnn_model.weights)
            optimizer.apply_gradients(zip(gradients, rnn_model.weights))

            if (epoch + 1) % 10 == 0:
                print("epoch %d, loss %f" % (epoch, loss))

    def test_basic_simplernn_dynamic_2(self):

        class CustomisedModel(tl.models.Model):
            def __init__(self):
                super(CustomisedModel, self).__init__()
                self.rnnlayer = tl.layers.RNN(
                    cell=tf.keras.layers.SimpleRNNCell(units=8, dropout=0.1),
                    inputs_shape=[2, 6, 4],
                    return_last=False, return_seq_2d=False, return_state=False, name='simplernn'
                )
                self.dense = tl.layers.Dense(in_channels=8, n_units=1)

            def forward(self, x):
                z = self.rnnlayer(x, return_seq_2d=True)
                z = self.dense(z[-2:,:])
                return z

        rnn_model = CustomisedModel()
        print(rnn_model)
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        rnn_model.train()
        assert rnn_model.rnnlayer.is_train

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y = rnn_model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, rnn_model.weights)
            optimizer.apply_gradients(zip(gradients, rnn_model.weights))

            if (epoch + 1) % 10 == 0:
                print("epoch %d, loss %f" % (epoch, loss))

    def test_basic_lstmrnn(self):

        inputs = tl.layers.Input([self.batch_size, self.num_steps, self.embedding_size])
        rnnlayer = tl.layers.RNN(
            cell=tf.keras.layers.LSTMCell(units=self.hidden_size, dropout=0.1),
            return_last=True, return_seq_2d=False, return_state=True, name='lstmrnn'
        )
        rnn, rnn_state = rnnlayer(inputs)
        outputs = tl.layers.Dense(n_units=1)(rnn)
        rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0], rnn_state[1]], name='rnn_model')
        print(rnn_model)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        rnn_model.train()

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y, final_h, final_c = rnn_model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, rnn_model.weights)
            optimizer.apply_gradients(zip(gradients, rnn_model.weights))

            if (epoch + 1) % 10 == 0:
                print("epoch %d, loss %f" % (epoch, loss))

    def test_basic_grurnn(self):

        inputs = tl.layers.Input([self.batch_size, self.num_steps, self.embedding_size])
        rnnlayer = tl.layers.RNN(
            cell=tf.keras.layers.GRUCell(units=self.hidden_size, dropout=0.1),
            return_last=True, return_seq_2d=False, return_state=True, name='grurnn'
        )
        rnn, rnn_state = rnnlayer(inputs)
        outputs = tl.layers.Dense(n_units=1)(rnn)
        rnn_model = tl.models.Model(inputs=inputs, outputs=[outputs, rnn_state[0]], name='rnn_model')
        print(rnn_model)

        optimizer = tf.optimizers.Adam(learning_rate=0.01)

        rnn_model.train()

        for epoch in range(50):
            with tf.GradientTape() as tape:
                pred_y, final_h = rnn_model(self.data_x)
                loss = tl.cost.mean_squared_error(pred_y, self.data_y)

            gradients = tape.gradient(loss, rnn_model.weights)
            optimizer.apply_gradients(zip(gradients, rnn_model.weights))

            if (epoch + 1) % 10 == 0:
                print("epoch %d, loss %f" % (epoch, loss))

    def test_sequence_length(self):
        data = [[[1],[2],[0],[0],[0]],
                [[1],[2],[3],[0],[0]],
                [[1],[2],[6],[1],[0]]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op(data)
        print(length)
        data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
                 [[2,3],[2,4],[3,2],[0,0],[0,0]],
                 [[3,3],[2,2],[5,3],[1,2],[0,0]]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op(data)
        print(length)

    def test_sequence_length2(self):
        data = [[1,2,0,0,0],
                [1,2,3,0,0],
                [1,2,6,1,0]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op2(data)
        print(length)

    def test_sequence_length3(self):
        data = [[[1],[2],[0],[0],[0]],
                [[1],[2],[3],[0],[0]],
                [[1],[2],[6],[1],[0]]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op3(data)
        print(length)
        data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
                [[2,3],[2,4],[3,2],[0,0],[0,0]],
                [[3,3],[2,2],[5,3],[1,2],[0,0]]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op3(data)
        print(length)
        data = [[1,2,0,0,0],
                [1,2,3,0,0],
                [1,2,6,1,0]]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        length = tl.layers.retrieve_seq_length_op3(data)
        print(length)
        data = [['hello','world','','',''],
                ['hello','world','tensorlayer','',''],
                ['hello','world','tensorlayer','2.0','']]
        data = tf.convert_to_tensor(data, dtype=tf.string)
        length = tl.layers.retrieve_seq_length_op3(data, pad_val='')
        print(length)

        try:
            data = [1,2,0,0,0]
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            length = tl.layers.retrieve_seq_length_op3(data)
            print(length)
        except Exception as e:
            print(e)

        try:
            data = np.random.random([4,2,6,2])
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            length = tl.layers.retrieve_seq_length_op3(data)
            print(length)
        except Exception as e:
            print(e)

if __name__ == '__main__':

    unittest.main()
