#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 TensorLayer. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example of Synced sequence input and output.

Generate text using LSTM.

Data: https://github.com/tensorlayer/tensorlayer/tree/master/example/data/

"""

import os
import re
import time

import numpy as np
import tensorflow as tf

import nltk
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)

_UNK = "_UNK"


def basic_clean_str(string):
    """Tokenization/string cleaning for a datasets."""
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", "  ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", "  ", string)
    string = re.sub(r"\"", "  ", string)  # "a"       --> " a "
    string = re.sub(r"\'", "  ", string)  # they'     --> they '
    string = re.sub(r"\’", "  ", string)  # they’     --> they ’
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\-", "  ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", "  ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", "  ", string)  # ( they)   --> ( they )
    string = re.sub(r"\]", "  ", string)  # they]     --> they ]
    string = re.sub(r"\[", "  ", string)  # they[     --> they [
    string = re.sub(r"\?", "  ", string)  # they?     --> they ?
    string = re.sub(r"\>", "  ", string)  # they>     --> they >
    string = re.sub(r"\<", "  ", string)  # they<     --> they <
    string = re.sub(r"\=", "  ", string)  # easier=   --> easier =
    string = re.sub(r"\;", "  ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", "  ", string)
    string = re.sub(r"\:", "  ", string)  # easier:   --> easier :
    string = re.sub(r"\"", "  ", string)  # easier"   --> easier "
    string = re.sub(r"\$", "  ", string)  # $380      --> $ 380
    string = re.sub(r"\_", "  ", string)  # _100     --> _ 100
    string = re.sub(r"\s{2,}", " ", string)  # Akara is    handsome --> Akara is handsome
    return string.strip().lower()  # lowercase


def customized_clean_str(string):
    """Tokenization/string cleaning for a datasets."""
    string = re.sub(r"\n", " ", string)  # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)  # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)  # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)  # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)  # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)  # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)  # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", " “ ", string)  # “a”       --> “ a ”
    string = re.sub(r"\”", " ” ", string)
    string = re.sub(r"\"", " “ ", string)  # "a"       --> " a "
    string = re.sub(r"\'", " ' ", string)  # they'     --> they '
    string = re.sub(r"\’", " ' ", string)  # they’     --> they '
    string = re.sub(r"\.", " . ", string)  # they.     --> they .
    string = re.sub(r"\,", " , ", string)  # they,     --> they ,
    string = re.sub(r"\-", " ", string)  # "low-cost"--> lost cost
    string = re.sub(r"\(", " ( ", string)  # (they)    --> ( they)
    string = re.sub(r"\)", " ) ", string)  # ( they)   --> ( they )
    string = re.sub(r"\!", " ! ", string)  # they!     --> they !
    string = re.sub(r"\]", " ] ", string)  # they]     --> they ]
    string = re.sub(r"\[", " [ ", string)  # they[     --> they [
    string = re.sub(r"\?", " ? ", string)  # they?     --> they ?
    string = re.sub(r"\>", " > ", string)  # they>     --> they >
    string = re.sub(r"\<", " < ", string)  # they<     --> they <
    string = re.sub(r"\=", " = ", string)  # easier=   --> easier =
    string = re.sub(r"\;", " ; ", string)  # easier;   --> easier ;
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"\:", " : ", string)  # easier:   --> easier :
    string = re.sub(r"\"", " \" ", string)  # easier"   --> easier "
    string = re.sub(r"\$", " $ ", string)  # $380      --> $ 380
    string = re.sub(r"\_", " _ ", string)  # _100     --> _ 100
    string = re.sub(r"\s{2,}", " ", string)  # Akara is    handsome --> Akara is handsome
    return string.strip().lower()  # lowercase


def customized_read_words(input_fpath):  # , dictionary):
    with open(input_fpath, "r", encoding="utf8") as f:
        words = f.read()
    # Clean the data
    words = customized_clean_str(words)
    # Split each word
    return words.split()


def main_restore_embedding_layer():
    """How to use Embedding layer, and how to convert IDs to vector,
    IDs to words, etc.
    """
    # Step 1: Build the embedding matrix and load the existing embedding matrix.
    vocabulary_size = 50000
    embedding_size = 128
    model_file_name = "model_word2vec_50k_128"
    batch_size = None

    if not os.path.exists(model_file_name + ".npy"):
        raise Exception(
            "Pretrained embedding matrix not found. "
            "Hint: Please pre-train the default model in "
            "`examples/text_word_embedding/tutorial_word2vec_basic.py`."
        )

    print("Load existing embedding matrix and dictionaries")
    all_var = tl.files.load_npy_to_any(name=model_file_name + '.npy')
    data = all_var['data']
    count = all_var['count']
    dictionary = all_var['dictionary']
    reverse_dictionary = all_var['reverse_dictionary']

    tl.nlp.save_vocab(count, name='vocab_' + model_file_name + '.txt')

    del all_var, data, count

    class Embedding_Model(Model):

        def __init__(self):
            super(Embedding_Model, self).__init__()
            self.embedding = Embedding(vocabulary_size, embedding_size)

        def forward(self, inputs):
            return self.embedding(inputs)

    model = Embedding_Model()
    model.eval()

    # TODO: assign certain parameters to model
    model.load_weights(model_file_name + ".hdf5", skip=True, in_order=False)

    # Step 2: Input word(s), output the word vector(s).
    word = 'hello'
    word_id = dictionary[word]
    print('word_id:', word_id)

    words = ['i', 'am', 'tensor', 'layer']
    word_ids = tl.nlp.words_to_word_ids(words, dictionary, _UNK)
    context = tl.nlp.word_ids_to_words(word_ids, reverse_dictionary)
    print('word_ids:', word_ids)
    print('context:', context)

    vector = model(word_id)
    print('vector:', vector.shape)
    print(vector)

    vectors = model(word_ids)
    print('vectors:', vectors.shape)
    print(vectors)


class Text_Generation_Net(Model):

    def __init__(self, vocab_size, hidden_size, init):
        super(Text_Generation_Net, self).__init__()

        self.embedding = Embedding(vocab_size, hidden_size, init, name='embedding')
        self.lstm = tl.layers.RNN(
            cell=tf.keras.layers.LSTMCell(hidden_size), return_last_output=False, return_last_state=True,
            return_seq_2d=True, in_channels=hidden_size
        )
        self.out_dense = Dense(vocab_size, in_channels=hidden_size, W_init=init, b_init=init, act=None, name='output')

    def forward(self, inputs, initial_state=None):
        embedding_vector = self.embedding(inputs)
        lstm_out, final_state = self.lstm(embedding_vector, initial_state=initial_state)
        logits = self.out_dense(lstm_out)
        return logits, final_state


def main_lstm_generate_text():
    """Generate text by Synced sequence input and output."""
    # rnn model and update  (describtion: see tutorial_ptb_lstm.py)
    init_scale = 0.1
    learning_rate = 1e-3
    sequence_length = 20
    hidden_size = 200
    max_epoch = 100
    batch_size = 16

    top_k_list = [1, 3, 5, 10]
    print_length = 30

    model_file_name = "model_generate_text.hdf5"

    #  ===== Prepare Data
    words = customized_read_words(input_fpath="data/trump/trump_text.txt")

    vocab = tl.nlp.create_vocab([words], word_counts_output_file='vocab.txt', min_word_count=1)
    vocab = tl.nlp.Vocabulary('vocab.txt', unk_word="<UNK>")
    vocab_size = vocab.unk_id + 1
    train_data = [vocab.word_to_id(word) for word in words]

    # Set the seed to generate sentence.
    seed = "it is a"
    # seed = basic_clean_str(seed).split()
    seed = nltk.tokenize.word_tokenize(seed)
    print('seed : %s' % seed)

    init = tl.initializers.random_uniform(-init_scale, init_scale)

    net = Text_Generation_Net(vocab_size, hidden_size, init)

    train_weights = net.trainable_weights
    optimizer = tf.optimizers.Adam(lr=learning_rate)

    # ===== Training

    print("\nStart learning a model to generate text")
    for i in range(max_epoch):

        print("Epoch: %d/%d" % (i + 1, max_epoch))
        epoch_size = ((len(train_data) // batch_size) - 1) // sequence_length

        start_time = time.time()
        costs = 0.0
        iters = 0

        net.train()
        # reset all states at the begining of every epoch
        lstm_state = None
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data, batch_size, sequence_length)):
            with tf.GradientTape() as tape:
                ## compute outputs
                logits, lstm_state = net(x, initial_state=lstm_state)
                ## compute loss and update model
                cost = tl.cost.cross_entropy(logits, tf.reshape(y, [-1]), name='train_loss')

            grad = tape.gradient(cost, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

            costs += cost
            iters += 1

            if step % (epoch_size // 10) == 1:
                print(
                    "%.3f perplexity: %.3f speed: %.0f wps" % (
                        step * 1.0 / epoch_size, np.exp(costs / iters),
                        iters * batch_size * sequence_length * batch_size / (time.time() - start_time)
                    )
                )
        train_perplexity = np.exp(costs / iters)
        # print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        print("Epoch: %d/%d Train Perplexity: %.3f" % (i + 1, max_epoch, train_perplexity))

        net.eval()
        # for diversity in diversity_list:
        # testing: sample from top k words
        for top_k in top_k_list:
            # Testing, generate some text from a given seed.
            lstm_state = None
            outs_id = [vocab.word_to_id(w) for w in seed]
            # feed the seed to initialize the state for generation.
            for ids in outs_id[:-1]:
                a_id = np.asarray(ids).reshape(1, 1)
                _, lstm_state = net(a_id, initial_state=lstm_state)

            # feed the last word in seed, and start to generate sentence.
            a_id = outs_id[-1]
            for _ in range(print_length):
                a_id = np.asarray(a_id).reshape(1, 1)
                logits, lstm_state = net(a_id, initial_state=lstm_state)
                out = tf.nn.softmax(logits)
                # Without sampling
                # a_id = np.argmax(out[0])
                # Sample from all words, if vocab_size is large,
                # this may have numeric error.
                # a_id = tl.nlp.sample(out[0], diversity)
                # Sample from the top k words.
                a_id = tl.nlp.sample_top(out[0].numpy(), top_k=top_k)
                outs_id.append(a_id)
            sentence = [vocab.id_to_word(w) for w in outs_id]
            sentence = " ".join(sentence)
            # print(diversity, ':', sentence)
            print(top_k, ':', sentence)

    print("Save model")
    net.save_weights(model_file_name)


if __name__ == '__main__':
    # Restore a pretrained embedding matrix
    # main_restore_embedding_layer()

    # How to generate text from a given context
    main_lstm_generate_text()
