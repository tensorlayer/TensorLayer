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

tf.nn.seq2seq
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py

Data
----
http://www.statmt.org/wmt10/

tensorflow (0.9.0)
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
  Corresponding source data and target data in the same line.

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
    """Step 1 : Download Training and Testing data.
    Compare with Word2vec example, the dataset in this example is large,
    so we use TensorFlow's gfile functions to speed up the pre-processing.
    """
    data_dir = "wmt"           # Data directory
    train_dir = "wmt"          # Model directory save_dir
    fr_vocab_size = 40000      # French vocabulary size
    en_vocab_size = 40000      # English vocabulary size

    ## Download or Load data
    train_path, dev_path = tl.files.load_wmt_en_fr_dataset(data_dir=data_dir)
    print("Training data : %s" % train_path)   # wmt/giga-fren.release2
    print("Testing data : %s" % dev_path)     # wmt/newstest2013

    """Step 2 : Create Vocabularies for both Training and Testing data.
    """
    ## Create vocabulary file (if it does not exist yet) from data file.
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])") # regular expression for word spliting. in basic_tokenizer.
    _DIGIT_RE = re.compile(br"\d")  # regular expression for search digits
    normalize_digits = True         # replace all digits to 0
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
    tl.nlp.create_vocabulary(fr_vocab_path, train_path + ".fr",
                fr_vocab_size, tokenizer=None, normalize_digits=normalize_digits,
                _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)
    tl.nlp.create_vocabulary(en_vocab_path, train_path + ".en",
                en_vocab_size, tokenizer=None, normalize_digits=normalize_digits,
                _DIGIT_RE=_DIGIT_RE, _START_VOCAB=_START_VOCAB)

    """ Step 3 : Tokenize Training and Testing data.
    """
    ## Create tokenized file for the training data by using the vocabulary file.
    # normalize_digits=True means set all digits to zero, so as to reduce
    # vocabulary size.
    fr_train_ids_path = train_path + (".ids%d.fr" % fr_vocab_size)
    en_train_ids_path = train_path + (".ids%d.en" % en_vocab_size)
    print("Tokenized Training data of French : %s" % fr_train_ids_path)    # wmt/giga-fren.release2.ids40000.fr
    print("Tokenized Training data of English : %s" % en_train_ids_path)   # wmt/giga-fren.release2.ids40000.fr
    tl.nlp.data_to_token_ids(train_path + ".fr", fr_train_ids_path, fr_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(train_path + ".en", en_train_ids_path, en_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    ## and also, we should create tokenized file for the development (testing) data.
    fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocab_size)
    en_dev_ids_path = dev_path + (".ids%d.en" % en_vocab_size)
    print("Tokenized Testing data of French : %s" % fr_dev_ids_path)    # wmt/newstest2013.ids40000.fr
    print("Tokenized Testing data of English : %s" % en_dev_ids_path)   # wmt/newstest2013.ids40000.en
    tl.nlp.data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.files.data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    ## You can get the word_to_id dictionary and id_to_word list as follow.
    # vocab, rev_vocab = tl.nlp.initialize_vocabulary(en_vocab_path)
    # print(vocab)
    # {b'cat': 1, b'dog': 0, b'bird': 2}
    # print(rev_vocab)
    # [b'dog', b'cat', b'bird']

    en_train = en_train_ids_path
    fr_train = fr_train_ids_path
    en_dev = en_dev_ids_path
    fr_dev = fr_dev_ids_path


    """Step 4 : Load both tokenized Training and Testing data into buckets
    and compute their size.
    """
    print()
    # Bucketing is a method to efficiently handle sentences of different length.
    # When translating English to French, we will have English sentences of
    # different lengths I on input, and French sentences of different
    # lengths O on output. We should in principle create a seq2seq model
    # for every pair (I, O+1) of lengths of an English and French sentence.
    #
    # For find the closest bucket for each pair, then we could just pad every
    # sentence with a special PAD symbol in the end if the bucket is bigger
    # than the sentence
    #
    # We use a number of buckets and pad to the closest one for efficiency.
    #
    # If the input is an English sentence with 3 tokens, and the corresponding
    # output is a French sentence with 6 tokens, then they will be put in the
    # first bucket and padded to length 5 for encoder inputs (English sentence),
    # and length 10 for decoder inputs.
    # If we have an English sentence with 8 tokens and the corresponding French
    # sentence has 18 tokens, then they will be fit into (20, 25) bucket.
    #
    # Given a pair [["I", "go", "."], ["Je", "vais", "."]] in tokenized format.
    # The training data of encoder inputs representing [PAD PAD "." "go" "I"]
    # and decoder inputs [GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD].
    # see ``get_batch()``
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

    num_layers = 3
    size = 1024
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 64
    num_samples = 512
    max_train_data_size = 0     # Limit on the size of training data (0: no limit). DH: for fast testing, set a value
    steps_per_checkpoint = 10        # Print, save frequence
    plot_data = True

    print ("Reading development (testing) data into buckets")
    dev_set = read_data(en_dev, fr_dev, buckets, EOS_ID)

    if plot_data:
        # Visualize the development (testing) data
        print('dev data:', buckets[0], dev_set[0][0])    # (5, 10), [[13388, 4, 949], [23113, 8, 910, 2]]
        vocab_en, rev_vocab_en = tl.nlp.initialize_vocabulary(en_vocab_path)
        context = tl.nlp.word_ids_to_words(dev_set[0][0][0], rev_vocab_en)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_en)
        print('en word_ids:', word_ids) # [13388, 4, 949]
        print('en context:', context)   # [b'Preventing', b'the', b'disease']
        vocab_fr, rev_vocab_fr = tl.nlp.initialize_vocabulary(fr_vocab_path)
        context = tl.nlp.word_ids_to_words(dev_set[0][0][1], rev_vocab_fr)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_fr)
        print('fr word_ids:', word_ids) # [23113, 8, 910, 2]
        print('fr context:', context)   # [b'Pr\xc3\xa9venir', b'la', b'maladie', b'_EOS']
        print()

    print ("Reading training data  into buckets (limit: %d)." % max_train_data_size)
    train_set = read_data(en_train, fr_train, buckets, EOS_ID, max_train_data_size)
    if plot_data:
        # Visualize the training data
        print('train data:', buckets[0], train_set[0][0])   # (5, 10) [[1368, 3344], [1089, 14, 261, 2]]
        context = tl.nlp.word_ids_to_words(train_set[0][0][0], rev_vocab_en)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_en)
        print('en word_ids:', word_ids) # [1368, 3344]
        print('en context:', context)   # [b'Site', b'map']
        context = tl.nlp.word_ids_to_words(train_set[0][0][1], rev_vocab_fr)
        word_ids = tl.nlp.words_to_word_ids(context, vocab_fr)
        print('fr word_ids:', word_ids) # [1089, 14, 261, 2]
        print('fr context:', context)   # [b'Plan', b'du', b'site', b'_EOS']
        print()

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    print('the num of training data in each buckets:', train_bucket_sizes)    # [239121, 1344322, 5239557, 10445326]
    print('the num of training data:', train_total_size)        # 17268326.0

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print('train_buckets_scale:',train_buckets_scale)   # [0.013847375825543252, 0.09169638099257565, 0.3951164693091849, 1.0]


    """Step 5 : Create a list of placeholders for different buckets
    """
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    # None is
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
        target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one (DH: remove the GO symbol)
    targets = [decoder_inputs[i + 1]
               for i in xrange(len(decoder_inputs) - 1)]

    # Give [["I", "go", "."], ["Je", "vais", "."]]  bucket = (I, O) = (5, 10)
    # encoder_inputs = [PAD PAD "." "go" "I"]                           <-- I
    # decoder_inputs = [GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD]     <-- O
    # targets        = ["Je" "vais" "." EOS PAD PAD PAD PAD PAD]        <-- O - 1
    # target_weights =                                                  <-- O
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
        # Use sampled softmax to handle large output vocabulary
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # If 0 < num_samples < target_vocab_size, we use sampled softmax,
        # Otherwise,
        # In this case, as target_vocab_size=4000, for vocabularies smaller
        # than 512, it might be a better idea to just use a standard softmax loss.
        if num_samples > 0 and num_samples < target_vocab_size:
            w = tf.get_variable("proj_w", [size, target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [target_vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels, num_samples, target_vocab_size):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                    target_vocab_size)
            softmax_loss_function = sampled_loss

        # network = tl.layers.RNNLayer(network,
        #             cell_fn=tf.nn.rnn_cell.GRUCell,
        #             cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
        #             n_hidden=size,
        #             initializer=tf.random_uniform_initializer(-0.1, 0.1),
        #             n_steps=num_steps,
        #             return_last=False,
        #             name='basic_lstm_layer1')



    model = inference(en_vocab_size, fr_vocab_size, buckets, size,
                                        num_layers, is_train=True)

    # loss and updates
    learning_rate = tf.Variable(float(learning_rate), trainable=False)
    learning_rate_decay_op = learning_rate.assign(
                    learning_rate * learning_rate_decay_factor)
    global_step = tf.Variable(0, trainable=False)



def self_test_dh():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    buckets = [(3, 3), (6, 6)]
    num_layers = 2
    hidden_size = 32
    source_vocab_size = target_vocab_size = 10
    max_gradient_norm = 5.0
    batch_size = 32
    learning_rate = 0.3
    learning_rate_decay_factor = 0.99
    num_samples = 8

    model = seq2seq_model.Seq2SeqModel(source_vocab_size, target_vocab_size,
                                buckets, hidden_size, num_layers,
                                max_gradient_norm, batch_size, learning_rate,
                                learning_rate_decay_factor, use_lstm=False,
                                num_samples=num_samples, forward_only=False)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([0, 1], [2, 2]), ([2, 3], [4]), ([5], [6])],
                [([1, 1, 2, 3, 4], [2, 2, 2, 2, 2]), ([3, 3, 4], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1]) # random choice bucket 0 or 1
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      print('what is the data')
      print(encoder_inputs, len(encoder_inputs), len(encoder_inputs[0]))
      print(decoder_inputs, len(decoder_inputs), len(decoder_inputs[0]))
      print(target_weights, len(target_weights), len(target_weights[0]))
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        main_train()
        # self_test_dh()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
