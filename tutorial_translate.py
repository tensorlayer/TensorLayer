#! /usr/bin/python
# -*- coding: utf8 -*-


# Copyright 2016 The TensoLayer Contributors. All Rights Reserved.
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

"""Sequence-to-sequence model with an attention mechanism for multiple buckets.
Translate English to French.

This example implements a multi-layer recurrent neural network as encoder,
and an attention-based decoder. This is the same as the model described in
this paper:
    “Grammar as a Foreign Language”
    http://arxiv.org/abs/1412.7449 - please look there for details,
or into the seq2seq library for complete model implementation.
This example also allows to use GRU cells in addition to LSTM cells, and
sampled softmax to handle large output vocabulary size. A single-layer
version of this model, but with bi-directional encoder, was presented in
    “Neural Machine Translation by Jointly Learning to Align and Translate”
    http://arxiv.org/abs/1409.0473
and sampled softmax is described in Section 3 of the following paper.
    “On Using Very Large Target Vocabulary for Neural Machine Translation”
    http://arxiv.org/abs/1412.2007

References
-----------
tensorflow/models/rnn/translate
https://www.tensorflow.org/versions/r0.9/tutorials/seq2seq/index.html

Data
----
http://www.statmt.org/wmt10/
"""

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import time
import os
import re
import sys
from six.moves import xrange


def read_data(source_path, target_path, buckets, EOS_ID, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def main_train():
    """Compare with Word2vec example, the dataset in this example is large,
    so we use TensorFlow's gfile functions to speed up the pre-processing.
    """
    data_dir = "wmt"           # Data directory
    fr_vocab_size = 40000      # French vocabulary size
    en_vocab_size = 40000      # English vocabulary size

    ## Download or Load data
    train_path, dev_path = tl.files.load_wmt_en_fr_dataset(data_dir=data_dir)
    print("Training data : %s" % train_path)   # wmt/giga-fren.release2
    print("Testing data : %s" % dev_path)     # wmt/newstest2013

    ## Create vocabulary file (if it does not exist yet) from data file.
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])") # regular expression for word spliting. in basic_tokenizer.
    _DIGIT_RE = re.compile(br"\d")  # regular expression for search digits
    # Special vocabulary symbols
    _PAD = b"_PAD"  #
    _GO = b"_GO"    # start to generate the output sentence
    _EOS = b"_EOS"  # end of sentence of the output sentence
    _UNK = b"_UNK"  # unknown word
    PAD_ID = 0      # index (row number) in vocabulary
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
    fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocab_size)
    en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocab_size)
    print("Vocabulary of French : %s" % fr_vocab_path)    # wmt/vocab40000.fr
    print("Vocabulary of English : %s" % en_vocab_path)   # wmt/vocab40000.en
    tl.files.create_vocabulary(fr_vocab_path, train_path + ".fr",
                    fr_vocab_size, tokenizer=None, normalize_digits=True,
                    _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)
    tl.files.create_vocabulary(en_vocab_path, train_path + ".en",
                    en_vocab_size, tokenizer=None, normalize_digits=True,
                    _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)

    ## Create tokenized file for the training data by using the vocabulary file.
    # normalize_digits=True means set all digits to zero, so as to reduce
    # vocabulary size.
    fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocab_size)
    en_train_ids_path = train_path + (".ids%d.en" % en_vocab_size)
    print("Tokenized Training data of French : %s" % fr_train_ids_path)    # wmt/giga-fren.release2.ids40000.fr
    print("Tokenized Training data of English : %s" % en_train_ids_path)   # wmt/giga-fren.release2.ids40000.fr
    tl.files.data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path,
                                    tokenizer=None, normalize_digits=True,
                                    UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.files.data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path,
                                    tokenizer=None, normalize_digits=True,
                                    UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    ## and also, we should create tokenized file for the development (testing) data.
    fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocab_size)
    en_dev_ids_path = dev_path + (".ids%d.en" % en_vocab_size)
    print("Tokenized Testing data of French : %s" % fr_dev_ids_path)    # wmt/newstest2013.ids40000.fr
    print("Tokenized Testing data of English : %s" % en_dev_ids_path)   # wmt/newstest2013.ids40000.en
    tl.files.data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path,
                                    tokenizer=None, normalize_digits=True,
                                    UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.files.data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path,
                                    tokenizer=None, normalize_digits=True,
                                    UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    # You can get the word_to_id dictionary and id_to_word list as follow.
    # vocab, rev_vocab = tl.files.initialize_vocabulary(en_vocab_path)
    # print(vocab)
    # {b'cat': 1, b'dog': 0, b'bird': 2}
    # print(rev_vocab)
    # [b'dog', b'cat', b'bird']

    en_train = en_train_ids_path
    fr_train = fr_train_ids_path
    en_dev = en_dev_ids_path
    fr_dev = fr_dev_ids_path



    """After download and tokenized the training/testing data and
    create the vocabularies for English and French.
    We can read the data into buckets and compute their size, and then
    build the model.
    """
    print()
    # We use a number of buckets and pad to the closest one for efficiency.
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

    num_layers = 3
    size = 1024
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 64
    num_samples = 512
    max_train_data_size = 100000     # Limit on the size of training data (0: no limit). DH: for fast testing, set a value
    steps_per_checkpoint = 200  # Print, save frequence

    ## Read all tokenized data into buckets and compute their sizes.
    # i.e. read all training data to memory.
    print ("Reading development (testing)  data")
    dev_set = read_data(en_dev, fr_dev, buckets, EOS_ID)

    # Visualize the development (testing) data
    print('dev data:', buckets[0], dev_set[0][0])    # (5, 10), [[13388, 4, 949], [23113, 8, 910, 2]]
    vocab_en, rev_vocab_en = tl.files.initialize_vocabulary(en_vocab_path)
    context = tl.files.word_ids_to_words(dev_set[0][0][0], rev_vocab_en)
    word_ids = tl.files.words_to_word_ids(context, vocab_en)
    print('en word_ids:', word_ids) # [13388, 4, 949]
    print('en context:', context)   # [b'Preventing', b'the', b'disease']
    vocab_fr, rev_vocab_fr = tl.files.initialize_vocabulary(fr_vocab_path)
    context = tl.files.word_ids_to_words(dev_set[0][0][1], rev_vocab_fr)
    word_ids = tl.files.words_to_word_ids(context, vocab_fr)
    print('fr word_ids:', word_ids) # [23113, 8, 910, 2]
    print('fr context:', context)   # [b'Pr\xc3\xa9venir', b'la', b'maladie', b'_EOS']
    print()
    # exit()

    print ("Reading training data (limit: %d)." % max_train_data_size)
    train_set = read_data(en_train, fr_train, buckets, EOS_ID, max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print('train_bucket_sizes:', train_bucket_sizes)    # [5807, 9719, 28191, 43465]    when max_train_data_size=100000
    print('train_total_size:', train_total_size)        # 87182                          when max_train_data_size=100000

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print('train_buckets_scale:',train_buckets_scale)   # [0.0666077860108738, 0.1780872198389576, 0.5014452524603703, 1.0]  when max_train_data_size=100000

    # Visualize the training data
    print('train data:', buckets[0], train_set[0][0])   # (5, 10) [[1368, 3344], [1089, 14, 261, 2]]
    context = tl.files.word_ids_to_words(train_set[0][0][0], rev_vocab_en)
    word_ids = tl.files.words_to_word_ids(context, vocab_en)
    print('en word_ids:', word_ids) # [1368, 3344]
    print('en context:', context)   # [b'Site', b'map']
    context = tl.files.word_ids_to_words(train_set[0][0][1], rev_vocab_fr)
    word_ids = tl.files.words_to_word_ids(context, vocab_fr)
    print('fr word_ids:', word_ids) # [1089, 14, 261, 2]
    print('fr context:', context)   # [b'Plan', b'du', b'site', b'_EOS']
    print()


    exit()
    # placeholders using buckets
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
        target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [decoder_inputs[i + 1]
               for i in xrange(len(decoder_inputs) - 1)]

    print(len(encoder_inputs))  # 40
    print(len(decoder_inputs))  # 51
    print(len(targets))         # 50
    print(len(target_weights))  # 51
    exit()


    # print("Creating %d layers of %d units." % (num_layers, size))

    # def(source_vocab_size, target_vocab_size, buckets, size,
    #          num_layers, max_gradient_norm, batch_size, learning_rate,
    #          learning_rate_decay_factor, use_lstm=False,
    #          num_samples=512, forward_only=False):
        #   model = seq2seq_model.Seq2SeqModel(
        #   FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
        #   FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
        #   FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
        #   forward_only=forward_only):
    def inference(source_vocab_size, target_vocab_size, buckets, size,
                                        num_layers, is_train=True):
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < target_vocab_size:
            w = tf.get_variable("proj_w", [size, target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [target_vocab_size])
            output_projection = (w, b)
            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                    target_vocab_size)
            softmax_loss_function = sampled_loss

        network = tl.layers.RNNLayer(network,
                    cell_fn=tf.nn.rnn_cell.GRUCell,
                    cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
                    n_hidden=size,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    n_steps=num_steps,
                    return_last=False,
                    name='basic_lstm_layer1')



    model = inference(en_vocab_size, fr_vocab_size, buckets, size,
                                        num_layers, is_train=True)

    # loss and updates
    learning_rate = tf.Variable(float(learning_rate), trainable=False)
    learning_rate_decay_op = learning_rate.assign(
                    learning_rate * learning_rate_decay_factor)
    global_step = tf.Variable(0, trainable=False)





if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        main_train()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
