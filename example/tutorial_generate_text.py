#! /usr/bin/python
# -*- coding: utf-8 -*-



# Copyright 2016 TensorLayer. All Rights Reserved.
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
# ==============================================================================

"""Example of Synced sequence input and output.
Generate text using LSTM.

Data: https://github.com/zsdonghao/tensorlayer/tree/master/example/data/

"""

import os
import re
import time

import nltk
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


# # _UNK = "_UNK"

def basic_clean_str(string):
    """Tokenization/string cleaning for a datasets.
    """
    string = re.sub(r"\n", " ", string)         # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)      # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)   # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)    # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)    # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)         # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)   # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", "  ", string)       # “a”       --> “ a ”
    string = re.sub(r"\”", "  ", string)
    string = re.sub(r"\"", "  ", string)       # "a"       --> " a "
    string = re.sub(r"\'", "  ", string)        # they'     --> they '
    string = re.sub(r"\’", "  ", string)        # they’     --> they ’
    string = re.sub(r"\.", " . ", string)       # they.     --> they .
    string = re.sub(r"\,", " , ", string)        # they,     --> they ,
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\-", "  ", string)        # "low-cost"--> lost cost
    string = re.sub(r"\(", "  ", string)       # (they)    --> ( they)
    string = re.sub(r"\)", "  ", string)       # ( they)   --> ( they )
    string = re.sub(r"\]", "  ", string)       # they]     --> they ]
    string = re.sub(r"\[", "  ", string)       # they[     --> they [
    string = re.sub(r"\?", "  ", string)       # they?     --> they ?
    string = re.sub(r"\>", "  ", string)       # they>     --> they >
    string = re.sub(r"\<", "  ", string)       # they<     --> they <
    string = re.sub(r"\=", "  ", string)        # easier=   --> easier =
    string = re.sub(r"\;", "  ", string)        # easier;   --> easier ;
    string = re.sub(r"\;", "  ", string)
    string = re.sub(r"\:", "  ", string)        # easier:   --> easier :
    string = re.sub(r"\"", "  ", string)      # easier"   --> easier "
    string = re.sub(r"\$", "  ", string)       # $380      --> $ 380
    string = re.sub(r"\_", "  ", string)        # _100     --> _ 100
    string = re.sub(r"\s{2,}", " ", string)     # Akara is    handsome --> Akara is handsome
    return string.strip().lower()               # lowercase

def customized_clean_str(string):
    """Tokenization/string cleaning for a datasets.
    """
    string = re.sub(r"\n", " ", string)         # '\n'      --> ' '
    string = re.sub(r"\'s", " \'s", string)      # it's      --> it 's
    string = re.sub(r"\’s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)   # they've   --> they have
    string = re.sub(r"\’ve", " have", string)
    string = re.sub(r"\'t", " not", string)    # can't     --> can not
    string = re.sub(r"\’t", " not", string)
    string = re.sub(r"\'re", " are", string)    # they're   --> they are
    string = re.sub(r"\’re", " are", string)
    string = re.sub(r"\'d", "", string)         # I'd (I had, I would) --> I
    string = re.sub(r"\’d", "", string)
    string = re.sub(r"\'ll", " will", string)   # I'll      --> I will
    string = re.sub(r"\’ll", " will", string)
    string = re.sub(r"\“", " “ ", string)       # “a”       --> “ a ”
    string = re.sub(r"\”", " ” ", string)
    string = re.sub(r"\"", " “ ", string)       # "a"       --> " a "
    string = re.sub(r"\'", " ' ", string)        # they'     --> they '
    string = re.sub(r"\’", " ' ", string)        # they’     --> they '
    string = re.sub(r"\.", " . ", string)       # they.     --> they .
    string = re.sub(r"\,", " , ", string)       # they,     --> they ,
    string = re.sub(r"\-", " ", string)         # "low-cost"--> lost cost
    string = re.sub(r"\(", " ( ", string)       # (they)    --> ( they)
    string = re.sub(r"\)", " ) ", string)       # ( they)   --> ( they )
    string = re.sub(r"\!", " ! ", string)       # they!     --> they !
    string = re.sub(r"\]", " ] ", string)       # they]     --> they ]
    string = re.sub(r"\[", " [ ", string)       # they[     --> they [
    string = re.sub(r"\?", " ? ", string)       # they?     --> they ?
    string = re.sub(r"\>", " > ", string)       # they>     --> they >
    string = re.sub(r"\<", " < ", string)       # they<     --> they <
    string = re.sub(r"\=", " = ", string)        # easier=   --> easier =
    string = re.sub(r"\;", " ; ", string)        # easier;   --> easier ;
    string = re.sub(r"\;", " ; ", string)
    string = re.sub(r"\:", " : ", string)        # easier:   --> easier :
    string = re.sub(r"\"", " \" ", string)      # easier"   --> easier "
    string = re.sub(r"\$", " $ ", string)       # $380      --> $ 380
    string = re.sub(r"\_", " _ ", string)        # _100     --> _ 100
    string = re.sub(r"\s{2,}", " ", string)     # Akara is    handsome --> Akara is handsome
    return string.strip().lower()               # lowercase

def customized_read_words(input_fpath):#, dictionary):
    with open(input_fpath, "r") as f:
        words = f.read()
    # Clean the data
    words = customized_clean_str(words)
    # Split each word
    return words.split()

def main_restore_embedding_layer():
    """How to use Embedding layer, and how to convert IDs to vector,
    IDs to words, etc.
    """
    ## Step 1: Build the embedding matrix and load the existing embedding matrix.
    vocabulary_size = 50000
    embedding_size = 128
    model_file_name = "model_word2vec_50k_128"
    batch_size = None

    print("Load existing embedding matrix and dictionaries")
    all_var = tl.files.load_npy_to_any(name=model_file_name+'.npy')
    data = all_var['data']; count = all_var['count']
    dictionary = all_var['dictionary']
    reverse_dictionary = all_var['reverse_dictionary']

    tl.nlp.save_vocab(count, name='vocab_'+model_file_name+'.txt')

    del all_var, data, count

    load_params = tl.files.load_npz(name=model_file_name+'.npz')

    x = tf.placeholder(tf.int32, shape=[batch_size])
    y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])

    emb_net = tl.layers.EmbeddingInputlayer(
                    inputs = x,
                    vocabulary_size = vocabulary_size,
                    embedding_size = embedding_size,
                    name ='embedding_layer')

    # sess.run(tf.initialize_all_variables())
    tl.layers.initialize_global_variables(sess)

    tl.files.assign_params(sess, [load_params[0]], emb_net)

    emb_net.print_params()
    emb_net.print_layers()

    ## Step 2: Input word(s), output the word vector(s).
    word = b'hello'
    word_id = dictionary[word]
    print('word_id:', word_id)

    words = [b'i', b'am', b'tensor', b'layer']
    word_ids = tl.nlp.words_to_word_ids(words, dictionary, _UNK)
    context = tl.nlp.word_ids_to_words(word_ids, reverse_dictionary)
    print('word_ids:', word_ids)
    print('context:', context)

    vector = sess.run(emb_net.outputs, feed_dict={x : [word_id]})
    print('vector:', vector.shape)

    vectors = sess.run(emb_net.outputs, feed_dict={x : word_ids})
    print('vectors:', vectors.shape)

def main_lstm_generate_text():
    """Generate text by Synced sequence input and output.
    """
    # rnn model and update  (describtion: see tutorial_ptb_lstm.py)
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    sequence_length = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 100
    lr_decay = 0.9
    batch_size = 20

    top_k_list = [1, 3, 5, 10]
    print_length = 30

    model_file_name = "model_generate_text.npz"

    ##===== Prepare Data
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

    sess = tf.InteractiveSession()

    ##===== Define model
    input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
    targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
    # Testing (Evaluation), for generate text
    input_data_test = tf.placeholder(tf.int32, [1, 1])

    def inference(x, is_train , sequence_length, reuse=None):
        """If reuse is True, the inferences use the existing parameters,
        then different inferences share the same parameters.
        """
        print("\nsequence_length: %d, is_train: %s, reuse: %s" %
                                            (sequence_length, is_train , reuse))
        rnn_init = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            network = EmbeddingInputlayer(
                        inputs=x,
                        vocabulary_size=vocab_size,
                        embedding_size=hidden_size,
                        E_init=rnn_init,
                        name='embedding')
            network = RNNLayer(network,
                        cell_fn=tf.contrib.rnn.BasicLSTMCell,
                        cell_init_args={'forget_bias': 0.0, 'state_is_tuple': True},
                        n_hidden=hidden_size,
                        initializer=rnn_init,
                        n_steps=sequence_length,
                        return_last=False,
                        return_seq_2d=True,
                        name='lstm1')
            lstm1 = network
            network = DenseLayer(network,
                        n_units=vocab_size,
                        W_init=rnn_init,
                        b_init=rnn_init,
                        act = tf.identity, name='output')
        return network, lstm1

    # Inference for Training
    network, lstm1 = inference(input_data,
                            is_train =True, sequence_length=sequence_length, reuse=None)
    # Inference for generate text, sequence_length=1
    network_test, lstm1_test = inference(input_data_test,
                            is_train=False, sequence_length=1, reuse=True)
    y_linear = network_test.outputs
    y_soft = tf.nn.softmax(y_linear)
    # y_id = tf.argmax(tf.nn.softmax(y), 1)

    ##===== Define train ops
    def loss_fn(outputs, targets, batch_size, sequence_length):
        # Returns the cost function of Cross-entropy of two sequences, implement
        # softmax internally.
        # outputs : 2D tensor [n_examples, n_outputs]
        # targets : 2D tensor [n_examples, n_outputs]
        # n_examples = batch_size * sequence_length
        # so
        # cost is the averaged cost of each mini-batch (concurrent process).
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [outputs],
            [tf.reshape(targets, [-1])],
            [tf.ones([batch_size * sequence_length])])
        cost = tf.reduce_sum(loss) / batch_size
        return cost

    ## Cost for Training
    cost = loss_fn(network.outputs, targets, batch_size, sequence_length)

    ## Truncated Backpropagation for training
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0.0, trainable=False)
    ## You can get all trainable parameters as follow.
    # tvars = tf.trainable_variables()
    ## Alternatively, you can specify the parameters for training as follw.
    #  tvars = network.all_params      $ all parameters
    #  tvars = network.all_params[1:]  $ parameters except embedding matrix
    ## Train the whole network.
    tvars = network.all_params
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))


    ##===== Training
    tl.layers.initialize_global_variables(sess)

    print("\nStart learning a model to generate text")
    for i in range(max_max_epoch):
        # decrease the learning_rate after ``max_epoch``, by multipling lr_decay.
        new_lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        sess.run(tf.assign(lr, learning_rate * new_lr_decay))

        print("Epoch: %d/%d Learning rate: %.8f" % (i + 1, max_max_epoch, sess.run(lr)))
        epoch_size = ((len(train_data) // batch_size) - 1) // sequence_length

        start_time = time.time()
        costs = 0.0; iters = 0
        ## reset all states at the begining of every epoch
        state1 = tl.layers.initialize_rnn_state(lstm1.initial_state)
        for step, (x, y) in enumerate(tl.iterate.ptb_iterator(train_data,
                                                    batch_size, sequence_length)):
            _cost, state1, _ = sess.run([cost,
                                    lstm1.final_state,
                                    train_op],
                                    feed_dict={input_data: x, targets: y,
                                        lstm1.initial_state: state1,
                                        })
            costs += _cost; iters += sequence_length

            if step % (epoch_size // 10) == 1:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / epoch_size, np.exp(costs / iters),
                    iters * batch_size / (time.time() - start_time)))
        train_perplexity = np.exp(costs / iters)
        # print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        print("Epoch: %d/%d Train Perplexity: %.3f" % (i + 1, max_max_epoch,
                                                            train_perplexity))

        # for diversity in diversity_list:
        ## testing: sample from top k words
        for top_k in top_k_list:
            # Testing, generate some text from a given seed.
            state1 = tl.layers.initialize_rnn_state(lstm1_test.initial_state)
            # state2 = tl.layers.initialize_rnn_state(lstm2_test.initial_state)
            outs_id = [vocab.word_to_id(w) for w in seed]
            # feed the seed to initialize the state for generation.
            for ids in outs_id[:-1]:
                a_id = np.asarray(ids).reshape(1,1)
                state1 = sess.run([lstm1_test.final_state,],
                                    feed_dict={input_data_test: a_id,
                                        lstm1_test.initial_state: state1,
                                        })
            # feed the last word in seed, and start to generate sentence.
            a_id = outs_id[-1]
            for _ in range(print_length):
                a_id = np.asarray(a_id).reshape(1,1)
                out, state1 = sess.run([y_soft,
                                    lstm1_test.final_state],
                                    feed_dict={input_data_test: a_id,
                                        lstm1_test.initial_state: state1,
                                        })
                ## Without sampling
                # a_id = np.argmax(out[0])
                ## Sample from all words, if vocab_size is large,
                # this may have numeric error.
                # a_id = tl.nlp.sample(out[0], diversity)
                ## Sample from the top k words.
                a_id = tl.nlp.sample_top(out[0], top_k=top_k)
                outs_id.append(a_id)
            sentence = [vocab.id_to_word(w) for w in outs_id]
            sentence = " ".join(sentence)
            # print(diversity, ':', sentence)
            print(top_k, ':', sentence)


    print("Save model")
    tl.files.save_npz(network_test.all_params, name=model_file_name)


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    """Restore a pretrained embedding matrix."""
    # main_restore_embedding_layer()
    """How to generate text from a given context."""
    main_lstm_generate_text()














#
