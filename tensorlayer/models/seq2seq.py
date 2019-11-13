#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from tensorlayer.models import Model

__all__ = ['Seq2seq']


class Seq2seq(Model):
    """vanilla stacked layer Seq2Seq model.

    Parameters
    ----------
    decoder_seq_length: int
        The length of your target sequence
    cell_enc : TensorFlow cell function
        The RNN function cell for your encoder stack, e.g tf.keras.layers.GRUCell
    cell_dec : TensorFlow cell function
        The RNN function cell for your decoder stack, e.g. tf.keras.layers.GRUCell
    n_layer : int
        The number of your RNN layers for both encoder and decoder block
    embedding_layer : tl.Layer
        A embedding layer, e.g. tl.layers.Embedding(vocabulary_size=voc_size, embedding_size=emb_dim)
    name : str
        The model name
    
    Examples
    ---------
    Classify stacked-layer Seq2Seq model, see `chatbot <https://github.com/tensorlayer/seq2seq-chatbot>`__

    Returns
    -------
        static stacked-layer Seq2Seq model.
    """

    def __init__(self, decoder_seq_length, cell_enc, cell_dec, n_units=256, n_layer=3, embedding_layer=None, name=None):
        super(Seq2seq, self).__init__(name=name)
        self.embedding_layer = embedding_layer
        self.vocabulary_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size
        self.n_layer = n_layer
        self.enc_layers = []
        self.dec_layers = []
        for i in range(n_layer):
            if (i == 0):
                self.enc_layers.append(
                    tl.layers.RNN(
                        cell=cell_enc(units=n_units), in_channels=self.embedding_size, return_last_state=True
                    )
                )
            else:
                self.enc_layers.append(
                    tl.layers.RNN(cell=cell_enc(units=n_units), in_channels=n_units, return_last_state=True)
                )

        for i in range(n_layer):
            if (i == 0):
                self.dec_layers.append(
                    tl.layers.RNN(
                        cell=cell_dec(units=n_units), in_channels=self.embedding_size, return_last_state=True
                    )
                )
            else:
                self.dec_layers.append(
                    tl.layers.RNN(cell=cell_dec(units=n_units), in_channels=n_units, return_last_state=True)
                )

        self.reshape_layer = tl.layers.Reshape([-1, n_units])
        self.dense_layer = tl.layers.Dense(n_units=self.vocabulary_size, in_channels=n_units)
        self.reshape_layer_after = tl.layers.Reshape([-1, decoder_seq_length, self.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape([-1, 1, self.vocabulary_size])

    def inference(self, encoding, seq_length, start_token, top_n):
        """Inference mode"""
        """
        Parameters
        ----------
        encoding : input tensor
            The source sequences 
        seq_length : int
            The expected length of your predicted sequence.
        start_token : int
            <SOS> : The token of "start of sequence"
        top_n : int
            Random search algorithm based on the top top_n words sorted by the probablity. 
        """
        feed_output = self.embedding_layer(encoding[0])
        state = [None for i in range(self.n_layer)]

        for i in range(self.n_layer):
            feed_output, state[i] = self.enc_layers[i](feed_output, return_state=True)
        batch_size = len(encoding[0].numpy())
        decoding = [[start_token] for i in range(batch_size)]
        feed_output = self.embedding_layer(decoding)
        for i in range(self.n_layer):
            feed_output, state[i] = self.dec_layers[i](feed_output, initial_state=state[i], return_state=True)

        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)
        feed_output = tf.argmax(feed_output, -1)
        # [B, 1]
        final_output = feed_output

        for i in range(seq_length - 1):
            feed_output = self.embedding_layer(feed_output)
            for i in range(self.n_layer):
                feed_output, state[i] = self.dec_layers[i](feed_output, initial_state=state[i], return_state=True)
            feed_output = self.reshape_layer(feed_output)
            feed_output = self.dense_layer(feed_output)
            feed_output = self.reshape_layer_individual_sequence(feed_output)
            ori_feed_output = feed_output
            if (top_n is not None):
                for k in range(batch_size):
                    idx = np.argpartition(ori_feed_output[k][0], -top_n)[-top_n:]
                    probs = [ori_feed_output[k][0][i] for i in idx]
                    probs = probs / np.sum(probs)
                    feed_output = np.random.choice(idx, p=probs)
                    feed_output = tf.convert_to_tensor([[feed_output]], dtype=tf.int64)
                    if (k == 0):
                        final_output_temp = feed_output
                    else:
                        final_output_temp = tf.concat([final_output_temp, feed_output], 0)
                feed_output = final_output_temp
            else:
                feed_output = tf.argmax(feed_output, -1)
            final_output = tf.concat([final_output, feed_output], 1)

        return final_output, state

    def forward(self, inputs, seq_length=20, start_token=None, return_state=False, top_n=None):

        state = [None for i in range(self.n_layer)]
        if (self.is_train):
            encoding = inputs[0]
            enc_output = self.embedding_layer(encoding)

            for i in range(self.n_layer):
                enc_output, state[i] = self.enc_layers[i](enc_output, return_state=True)

            decoding = inputs[1]
            dec_output = self.embedding_layer(decoding)

            for i in range(self.n_layer):
                dec_output, state[i] = self.dec_layers[i](dec_output, initial_state=state[i], return_state=True)

            dec_output = self.reshape_layer(dec_output)
            denser_output = self.dense_layer(dec_output)
            output = self.reshape_layer_after(denser_output)
        else:
            encoding = inputs
            output, state = self.inference(encoding, seq_length, start_token, top_n)

        if (return_state):
            return output, state
        else:
            return output
