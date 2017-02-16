#! /usr/bin/python
# -*- coding: utf8 -*-


# Copyright 2016 The TensorLayer Contributors. All Rights Reserved.
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

"""
from __future__ import print_function
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
import random
import math
import time
import os
import re
import sys
from six.moves import xrange

# Data directory and vocabularies size
data_dir = "wmt"                # Data directory
train_dir = "wmt"               # Model directory save_dir
fr_vocab_size = 40000           # French vocabulary size
en_vocab_size = 40000           # English vocabulary size
# Create vocabulary file (if it does not exist yet) from data file.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])") # regular expression for word spliting. in basic_tokenizer.
_DIGIT_RE = re.compile(br"\d")  # regular expression for search digits
normalize_digits = True         # replace all digits to 0
# Special vocabulary symbols
_PAD = b"_PAD"                  # Padding
_GO = b"_GO"                    # start to generate the output sentence
_EOS = b"_EOS"                  # end of sentence of the output sentence
_UNK = b"_UNK"                  # unknown word
PAD_ID = 0                      # index (row number) in vocabulary
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
plot_data = True
# Model
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
num_layers = 3
size = 1024
# Training
learning_rate = 0.5
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0             # Truncated backpropagation
batch_size = 64
num_samples = 512                   # Sampled softmax
max_train_data_size = 100             # Limit on the size of training data (0: no limit). DH: for fast testing, set a value
steps_per_checkpoint = 10           # Print, save frequence
# Save model
model_file_name = "model_translate_enfr"
resume = False
is_npz = False                     # if true save by npz file, otherwise ckpt file


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
    print()
    print("Prepare the raw data")
    train_path, dev_path = tl.files.load_wmt_en_fr_dataset(path=data_dir)
    print("Training data : %s" % train_path)   # wmt/giga-fren.release2
    print("Testing data : %s" % dev_path)     # wmt/newstest2013

    """Step 2 : Create Vocabularies for both Training and Testing data.
    """
    print()
    print("Create vocabularies")
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
    print()
    print("Tokenize data")
    # Create tokenized file for the training data by using the vocabulary file.
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

    # we should also create tokenized file for the development (testing) data.
    fr_dev_ids_path = dev_path + (".ids%d.fr" % fr_vocab_size)
    en_dev_ids_path = dev_path + (".ids%d.en" % en_vocab_size)
    print("Tokenized Testing data of French : %s" % fr_dev_ids_path)    # wmt/newstest2013.ids40000.fr
    print("Tokenized Testing data of English : %s" % en_dev_ids_path)   # wmt/newstest2013.ids40000.en
    tl.nlp.data_to_token_ids(dev_path + ".fr", fr_dev_ids_path, fr_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)
    tl.nlp.data_to_token_ids(dev_path + ".en", en_dev_ids_path, en_vocab_path,
                                tokenizer=None, normalize_digits=normalize_digits,
                                UNK_ID=UNK_ID, _DIGIT_RE=_DIGIT_RE)

    # You can get the word_to_id dictionary and id_to_word list as follow.
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

    Bucketing is a method to efficiently handle sentences of different length.
    When translating English to French, we will have English sentences of
    different lengths I on input, and French sentences of different
    lengths O on output. We should in principle create a seq2seq model
    for every pair (I, O+1) of lengths of an English and French sentence.

    For find the closest bucket for each pair, then we could just pad every
    sentence with a special PAD symbol in the end if the bucket is bigger
    than the sentence

    We use a number of buckets and pad to the closest one for efficiency.

    If the input is an English sentence with 3 tokens, and the corresponding
    output is a French sentence with 6 tokens, then they will be put in the
    first bucket and padded to length 5 for encoder inputs (English sentence),
    and length 10 for decoder inputs.
    If we have an English sentence with 8 tokens and the corresponding French
    sentence has 18 tokens, then they will be fit into (20, 25) bucket.

    Given a pair [["I", "go", "."], ["Je", "vais", "."]] in tokenized format.
    The training data of encoder inputs representing [PAD PAD "." "go" "I"]
    and decoder inputs [GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD].
    see ``get_batch()``
    """
    print()
    print ("Read development (test) data into buckets")
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
    print ("Read training data into buckets (limit: %d)" % max_train_data_size)
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
    print('the num of training data in each buckets: %s' % train_bucket_sizes)    # [239121, 1344322, 5239557, 10445326]
    print('the num of training data: %d' % train_total_size)        # 17268326.0

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print('train_buckets_scale:',train_buckets_scale)   # [0.013847375825543252, 0.09169638099257565, 0.3951164693091849, 1.0]


    """Step 6 : Create model
    """
    print()
    print("Create Embedding Attention Seq2seq Model")
    with tf.variable_scope("model", reuse=None):
        model = tl.layers.EmbeddingAttentionSeq2seqWrapper(
                          en_vocab_size,
                          fr_vocab_size,
                          buckets,
                          size,
                          num_layers,
                          max_gradient_norm,
                          batch_size,
                          learning_rate,
                          learning_rate_decay_factor,
                          forward_only=False)    # is_train = True

    # sess.run(tf.initialize_all_variables())
    tl.layers.initialize_global_variables(sess)
    # model.print_params()
    tl.layers.print_all_variables()

    if resume:
        print("Load existing model" + "!"*10)
        if is_npz:
            # instead of using TensorFlow saver, we can use TensorLayer to restore a model
            load_params = tl.files.load_npz(name=model_file_name+'.npz')
            tl.files.assign_params(sess, load_params, model)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_file_name+'.ckpt')

    """Step 7 : Training
    """
    print()
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        # randomly pick ``batch_size`` training examples from a random bucket_id
        # the data format is described in readthedocs tutorial
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)

        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / steps_per_checkpoint
        loss += step_loss / steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                            step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)

            # Save model
            if is_npz:
                tl.files.save_npz(model.all_params, name=model_file_name+'.npz')
            else:
                print('Model is saved to: %s' % model_file_name+'.ckpt')
                checkpoint_path = os.path.join(train_dir, model_file_name+'.ckpt')
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(buckets)):
                if len(dev_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % (bucket_id))
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)
                _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, True)
                eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()


def main_decode():
    # Create model and load parameters.
    with tf.variable_scope("model", reuse=None):
        model_eval = tl.layers.EmbeddingAttentionSeq2seqWrapper(
                      source_vocab_size = en_vocab_size,
                      target_vocab_size = fr_vocab_size,
                      buckets = buckets,
                      size = size,
                      num_layers = num_layers,
                      max_gradient_norm = max_gradient_norm,
                      batch_size = 1,  # We decode one sentence at a time.
                      learning_rate = learning_rate,
                      learning_rate_decay_factor = learning_rate_decay_factor,
                      forward_only = True) # is_train = False

    sess.run(tf.initialize_all_variables())

    if is_npz:
        print("Load parameters from npz")
        # instead of using TensorFlow saver, we can use TensorLayer to restore a model
        load_params = tl.files.load_npz(name=model_file_name+'.npz')
        tl.files.assign_params(sess, load_params, model_eval)
    else:
        print("Load parameters from ckpt")
        # saver = tf.train.Saver()
        # saver.restore(sess, model_file_name+'.ckpt')
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model_eval.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("no %s exist" % model_checkpoint_path)

    # model_eval.print_params()
    tl.layers.print_all_variables()

    # Load vocabularies.
    en_vocab_path = os.path.join(data_dir, "vocab%d.en" % en_vocab_size)
    fr_vocab_path = os.path.join(data_dir, "vocab%d.fr" % fr_vocab_size)
    en_vocab, _ = tl.nlp.initialize_vocabulary(en_vocab_path)
    _, rev_fr_vocab = tl.nlp.initialize_vocabulary(fr_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = tl.nlp.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(buckets))
                       if buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model_eval.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)
      # Get output logits for the sentence.
      _, _, output_logits = model_eval.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def main_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    buckets = [(3, 3), (6, 6)]
    num_layers = 2
    hidden_size = 32
    source_vocab_size = target_vocab_size = 10
    max_gradient_norm = 5.0
    batch_size = 3
    learning_rate = 0.3
    learning_rate_decay_factor = 0.99
    num_samples = 8

    model = tl.layers.EmbeddingAttentionSeq2seqWrapper(
                                source_vocab_size,
                                target_vocab_size,
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
          data_set, bucket_id, PAD_ID, GO_ID, EOS_ID, UNK_ID)
      print('what is the data')
      print(encoder_inputs, len(encoder_inputs), len(encoder_inputs[0]))
      print(decoder_inputs, len(decoder_inputs), len(decoder_inputs[0]))
      print(target_weights, len(target_weights), len(target_weights[0]))
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    try:
        """ Train model """
        main_train()
        """ Play with model """
        # main_decode()
        """ Quick test to see data format """
        main_test()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
