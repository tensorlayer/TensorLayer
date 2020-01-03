#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.models.transformer import Transformer
from tests.utils import CustomTestCase
from tensorlayer.models.transformer.utils import metrics
from tensorlayer.models.transformer.utils import attention_visualisation
import time


class TINY_PARAMS(object):
    vocab_size = 50+2
    encoder_num_layers = 2
    decoder_num_layers = 2
    hidden_size = 64
    ff_size = 16
    num_heads = 4
    keep_prob = 0.9

    # Default prediction params
    extra_decode_length = 5
    beam_size = 1
    alpha = 0.6  # used to calculate length normalization in beam search

    eos_id = 51
    sos_id = 0


class Model_Transformer_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 50

        cls.embedding_size = 32
        cls.dec_seq_length = 5
        cls.trainX = np.random.randint(low=0, high=50, size=(50, 11))
        cls.trainY = np.random.randint(low=0, high=50, size=(50, 10))

        cls.trainX[:, -1] = 51
        cls.trainY[:, -1] = 51
        # Parameters
        cls.src_len = len(cls.trainX)
        cls.tgt_len = len(cls.trainY)

        assert cls.src_len == cls.tgt_len

        cls.num_epochs = 100
        cls.n_step = cls.src_len // cls.batch_size

    @classmethod
    def tearDownClass(cls):
        pass

    def test_basic_simpleSeq2Seq(self):

        model_ = Transformer(TINY_PARAMS)

        # print(", ".join(x for x in [t.name for t in model_.trainable_weights]))

        self.vocab_size = TINY_PARAMS.vocab_size
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        for epoch in range(self.num_epochs):
            model_.train()
            t = time.time()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                with tf.GradientTape() as tape:

                    targets = Y
                    logits, weights_encoder, weights_decoder = model_(inputs=X, targets=Y)
                    logits = metrics.MetricLayer(self.vocab_size)([logits, targets])
                    logits, loss = metrics.LossLayer(self.vocab_size, 0.1)([logits, targets])

                    grad = tape.gradient(loss, model_.all_weights)
                    optimizer.apply_gradients(zip(grad, model_.all_weights))

                total_loss += loss
                n_iter += 1
            print(time.time() - t)
            tl.files.save_npz(model_.all_weights, name='./model_v4.npz')
            model_.eval()
            test_sample = trainX[0:2, :]
            model_.eval()
            [prediction, weights_decoder], weights_encoder = model_(inputs=test_sample)

            print("Prediction: >>>>>  ", prediction["outputs"], "\n Target: >>>>>  ", trainY[0:2, :], "\n\n")

            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))

        # visualise the self-attention weights at encoder during training
        trainX, trainY = shuffle(self.trainX, self.trainY)
        X = [trainX[0]]
        Y = [trainY[0]]
        logits, weights_encoder, weights_decoder = model_(inputs=X, targets=Y)
        attention_visualisation.plot_attention_weights(weights_encoder["layer_0"], X[0].numpy(), X[0].numpy())

        # visualise the encoder-decoder-attention weights at decoder during training
        trainX, trainY = shuffle(self.trainX, self.trainY)
        X = [trainX[0]]
        Y = [trainY[0]]
        logits, weights_encoder, weights_decoder = model_(inputs=X, targets=Y)
        attention_visualisation.plot_attention_weights(weights_decoder["enc_dec"]["layer_0"], X[0].numpy(), Y[0])

        # visualise the encoder-decoder-attention weights at decoder during inference
        trainX, trainY = shuffle(self.trainX, self.trainY)
        X = [trainX[0]]
        # Y = [trainY[0]]
        model_.eval()
        [prediction, weights_decoder], weights_encoder = model_(inputs=X)
        # print(X[0].numpy(), prediction["outputs"][0].numpy())
        attention_visualisation.plot_attention_weights(
            weights_decoder["enc_dec"]["layer_0"], X[0].numpy(), prediction["outputs"][0].numpy()
        )


if __name__ == '__main__':
    unittest.main()
