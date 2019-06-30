#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model
from tensorflow.python.ops.rnn_cell import LSTMCell
import numpy as np

from tests.utils import CustomTestCase


class LayerNode_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_net1(self):
        print('-' * 20, 'test_net1', '-' * 20)

        def get_model(input_shape):
            ni = Input(input_shape)

            nii = Conv2d(32, filter_size=(3, 3), strides=(1, 1), name='conv1')(ni)
            nn = Dropout(keep=0.9, name='drop1')(nii)

            conv = Conv2d(32, filter_size=(3, 3), strides=(1, 1), name='conv2')
            tt = conv(nn)  # conv2_node_0
            nn = conv(nn)  # conv2_node_1

            # a branch
            na = Conv2d(64, filter_size=(3, 3), strides=(1, 1), name='conv3')(nn)
            na = MaxPool2d(name='pool1')(na)

            # b branch
            nb = MaxPool2d(name='pool2')(nn)
            nb = conv(nb)  # conv2_node_2

            out = Concat(name='concat')([na, nb])
            M = Model(inputs=ni, outputs=[out, nn, nb])

            gg = conv(nii)  # this node will not be added since model fixed

            return M

        net = get_model([None, 24, 24, 3])

        for k, v in enumerate(net._node_by_depth):
            print(k, [x.name for x in v], [x.in_tensors_idxes for x in v])

        all_node_names = []
        for k, v in enumerate(net._node_by_depth):
            all_node_names.extend([x.name for x in v])

        self.assertNotIn('conv2_node_0', all_node_names)
        self.assertNotIn('conv2_node_3', all_node_names)

        self.assertEqual(len(net.all_layers), 8)
        print(net.all_layers)

        data = np.random.normal(size=[2, 24, 24, 3]).astype(np.float32)
        out, nn, nb = net(data, is_train=True)

        self.assertEqual(nn.shape, [2, 24, 24, 32])
        self.assertEqual(nb.shape, [2, 12, 12, 32])

    def test_net2(self):
        print('-' * 20, 'test_net2', '-' * 20)

        def get_unstack_model(input_shape):
            ni = Input(input_shape)

            nn = Dropout(keep=0.9)(ni)

            a, b, c = UnStack(axis=-1)(nn)

            b = Flatten()(b)
            b = Dense(10)(b)

            c = Flatten()(c)

            M = Model(inputs=ni, outputs=[a, b, c])
            return M

        net = get_unstack_model([None, 24, 24, 3])

        for k, v in enumerate(net._node_by_depth):
            print(k, [x.name for x in v], [x.in_tensors_idxes for x in v])

        data = np.random.normal(size=[2, 24, 24, 3]).astype(np.float32)
        out = net(data, is_train=True)

        self.assertEqual(len(out), 3)

    def test_word2vec(self):
        print('-' * 20, 'test_word2vec', '-' * 20)

        def get_word2vec():
            vocabulary_size = 800
            batch_size = 10
            embedding_size = 60
            num_sampled = 25
            inputs = tl.layers.Input([batch_size], dtype=tf.int32)
            labels = tl.layers.Input([batch_size, 1], dtype=tf.int32)

            emb_net = tl.layers.Word2vecEmbedding(
                vocabulary_size=vocabulary_size,
                embedding_size=embedding_size,
                num_sampled=num_sampled,
                activate_nce_loss=True,  # nce loss is activated
                nce_loss_args={},
                E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
                nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))),
                nce_b_init=tl.initializers.constant(value=0.0),
                name='word2vec_layer',
            )
            emb, nce = emb_net([inputs, labels])

            model = tl.models.Model(inputs=[inputs, labels], outputs=[emb, nce])
            return model

        net = get_word2vec()

        for k, v in enumerate(net._node_by_depth):
            print(k, [x.name for x in v], [x.in_tensors_idxes for x in v])

        x = tf.ones(shape=(10, ), dtype=tf.int32)
        y = tf.ones(shape=(10, 1), dtype=tf.int32)
        out = net([x, y], is_train=True)

        self.assertEqual(len(out), 2)

    def test_layerlist(self):
        print('-' * 20, 'layerlist', '-' * 20)

        class MyModel(Model):

            def __init__(self):
                super(MyModel, self).__init__()
                self.layers = LayerList([Dense(50, in_channels=100), Dropout(0.9), Dense(10, in_channels=50)])

            def forward(self, x):
                return self.layers(x)

        net = MyModel()
        self.assertEqual(net._nodes_fixed, False)

        data = np.random.normal(size=[4, 100]).astype(np.float32)
        out = net(data, is_train=False)

        self.assertEqual(net._nodes_fixed, True)
        self.assertEqual(net.layers._nodes_fixed, True)
        self.assertEqual(net.layers[0]._nodes_fixed, True)
        self.assertEqual(net.layers[1]._nodes_fixed, True)
        self.assertEqual(net.layers[2]._nodes_fixed, True)

    def test_ModelLayer(self):
        print('-' * 20, 'ModelLayer', '-' * 20)

        def MyModel():
            nii = Input(shape=[None, 100])
            nn = Dense(50, in_channels=100)(nii)
            nn = Dropout(0.9)(nn)
            nn = Dense(10)(nn)
            M = Model(inputs=nii, outputs=nn)
            return M

        mlayer = MyModel().as_layer()

        ni = Input(shape=[None, 100])
        nn = mlayer(ni)
        nn = Dense(5)(nn)
        net = Model(inputs=ni, outputs=nn)

        self.assertEqual(net._nodes_fixed, True)

        data = np.random.normal(size=[4, 100]).astype(np.float32)
        out = net(data, is_train=False)

        self.assertEqual(net._nodes_fixed, True)
        self.assertEqual(net.all_layers[1]._nodes_fixed, True)
        self.assertEqual(net.all_layers[1].model._nodes_fixed, True)
        self.assertEqual(net.all_layers[1].model.all_layers[0]._nodes_fixed, True)

    def test_STN(self):
        print('-' * 20, 'test STN', '-' * 20)

        def get_model(inputs_shape):
            ni = Input(inputs_shape)

            ## 1. Localisation network
            # use MLP as the localisation net
            nn = Flatten()(ni)
            nn = Dense(n_units=20, act=tf.nn.tanh)(nn)
            nn = Dropout(keep=0.8)(nn)
            # you can also use CNN instead for MLP as the localisation net

            ## 2. Spatial transformer module (sampler)
            stn = SpatialTransformer2dAffine(out_size=(40, 40), in_channels=20)
            # s = stn((nn, ni))
            nn = stn((nn, ni))
            s = nn

            ## 3. Classifier
            nn = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME')(nn)
            nn = Conv2d(16, (3, 3), (2, 2), act=tf.nn.relu, padding='SAME')(nn)
            nn = Flatten()(nn)
            nn = Dense(n_units=1024, act=tf.nn.relu)(nn)
            nn = Dense(n_units=10, act=tf.identity)(nn)

            M = Model(inputs=ni, outputs=[nn, s])
            return M

        net = get_model([None, 40, 40, 1])

        inputs = np.random.randn(2, 40, 40, 1).astype(np.float32)
        o1, o2 = net(inputs, is_train=True)
        self.assertEqual(o1.shape, (2, 10))
        self.assertEqual(o2.shape, (2, 40, 40, 1))

        self.assertEqual(len(net._node_by_depth), 10)


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
