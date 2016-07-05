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

"""
Vector Representations of Words
---------------------------------
This is the minimalistic implementation in
tensorflow/examples/tutorials/word2vec/word2vec_basic.py   (reimplementation)
This basic example contains the code needed to download some data,
train on it a bit and visualize the result by using t-SNE.

Once you get comfortable with reading and running the basic version,
you can graduate to
tensorflow/models/embedding/word2vec.py
which is a more serious implementation that showcases some more advanced
TensorFlow principles about how to efficiently use threads to move data
into a text model, how to checkpoint during training, etc.

If your model is no longer I/O bound but you want still more performance, you
can take things further by writing your own TensorFlow Ops, as described in
Adding a New Op. Again we've provided an example of this for the Skip-Gram case
tensorflow/models/embedding/word2vec_optimized.py.

Link
------
https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html#vector-representations-of-words

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
# import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorlayer as tl
import time



def main_word2vec_basic():
    # vanilla setting:
    # vocabulary_size = 50000 # maximum number of word in vocabulary
    # batch_size = 128
    # embedding_size = 128  # Dimension of the embedding vector (hidden layer).
    # skip_window = 1       # How many words to consider left and right.
    # num_skips = 2         # How many times to reuse an input to generate a label.
    #                       #     (should be double of 'skip_window' so as to
    #                       #     use both left and right words)
    # num_sampled = 64      # Number of negative examples to sample.
    # learning_rate = 1.0
    # num_steps = 100001    # total number of iteration

    # optimized setting:
    vocabulary_size = 200000 # maximum number of word in vocabulary
    batch_size = 20
    embedding_size = 200
    skip_window = 5
    num_skips = 10
    num_sampled = 100
    learning_rate = 0.2
    num_steps = 300001

    model_file_name = "model_word2vec.ckpt"
    resume = True  # load existing .ckpt

    """ Step 1: Download the data, read the context into a list of strings.
    """
    words = tl.files.load_matt_mahoney_text8_dataset()
    print('Data size', len(words)) # print(words)    # b'their', b'families', b'who', b'were', b'expelled', b'from', b'jerusalem',

    """ Step 2: Build the dictionary and replace rare words with 'UNK' token.
    """
    data, count, dictionary, reverse_dictionary = \
                        tl.files.build_words_dataset(words, vocabulary_size, True)
    print('Most 5 common words (+UNK)', count[:5]) # [['UNK', 418391], (b'the', 1061396), (b'of', 593677), (b'and', 416629), (b'one', 411764)]
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]]) # [5243, 3081, 12, 6, 195, 2, 3135, 46, 59, 156] [b'anarchism', b'originated', b'as', b'a', b'term', b'of', b'abuse', b'first', b'used', b'against']
    # save vocabulary to txt
    tl.files.save_vocab(count, name='vocab_text8.txt')

    del words  # Hint to reduce memory.
    del count

    """ Step 3: Function to generate a training batch for the Skip-Gram model.
    """
    data_index = 0
    batch, labels, data_index = tl.nlp.generate_skip_gram_batch(data=data,
                        batch_size=20, num_skips=4, skip_window=2, data_index=0)
    for i in range(20):
        print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


    """ Step 4: Build and train a Skip-Gram model.
    """
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        # a list of 'valid_size' integers smaller than 'valid_window'
        # print(valid_examples)   # [90 85 20 33 35 62 37 63 88 38 82 58 83 59 48 64]
    print_freq = 2000
        # n_epoch = int(num_steps / batch_size)
    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    # train_inputs is a row vector, a input is an integer id of single word.
    # train_labels is a column vector, a label is an integer id of single word.
    # valid_dataset is a column vector, a valid set is an integer id of single word.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Look up embeddings for inputs.
    # Note: a row of 'embeddings' is the vector representation of a word.
    # for the sake of speed, it is better to slice the embedding matrix
    # instead of transfering a word id to one-hot-format vector and then
    # multiply by the embedding matrix.
    # embed is the outputs of the hidden layer (embedding layer), it is a
    # row vector with 'embedding_size' values.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss (i.e. negative sampling)
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels
    # each time we evaluate the loss.
    cost = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                       inputs=embed, labels=train_labels,
                       num_sampled=num_sampled, num_classes=vocabulary_size,
                       num_true=1))
    # num_sampled: An int. The number of classes to randomly sample per batch
    #              Number of negative examples to sample.
    # num_classes: An int. The number of possible classes.
    # num_true = 1: An int. The number of target classes per training example.
    #            DH: if 1, predict one word given one word, like bigram model?  Check!

    # Construct the optimizer. Note: AdamOptimizer is very slow in this case
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # For simple visualization of validation set.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        # print(norm) # shape=(50000, 1)
    normalized_embeddings = embeddings / norm
    # Equal To: normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)
    valid_embed = tf.nn.embedding_lookup(
                                normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embed, normalized_embeddings, transpose_b=True)
        # multiply all valid word vector with all word vector.
        # transpose_b=True, normalized_embeddings is transposed before multiplication.

    """ Step 5: Begin training.
    """
    sess.run(tf.initialize_all_variables())
    if resume:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    average_loss = 0

    for step in xrange(num_steps):
        start_time = time.time()
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
                        data=data, batch_size=batch_size, num_skips=num_skips,
                        skip_window=skip_window, data_index=data_index)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
        # We perform one update step by evaluating the train_op (including it
        # in the list of returned values for sess.run()
        _, loss_val = sess.run([train_op, cost], feed_dict=feed_dict)
        average_loss += loss_val

        if step % print_freq == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step %d/%d. loss:%f took:%fs" %
                        (step, num_steps, average_loss, time.time() - start_time))
            average_loss = 0
        # Prints out nearby words given a list of words.
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % (print_freq * 5) == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors to print
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
        if step % (print_freq * 20) == 0:
            print("Save model " + "!"*10);
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)
    final_embeddings = normalized_embeddings.eval()

    """ Step 6: Visualize the embeddings.
    """
    tl.visualize.tsne_embedding(final_embeddings, reverse_dictionary,
                plot_only=500, second=5, saveable=True, name='word2vec_basic')


    """ Step 7: Evaluate by analogy questions.
    """
    #   from tensorflow/models/embedding/word2vec.py
    analogy_questions = tl.files.read_analogies_file( \
                eval_file='questions-words.txt', word2id=dictionary)
    # print(analogy_questions)
    # print(analogy_questions[0][0])
    # print(reverse_dictionary[analogy_questions[0][0]])
    # print(dictionary[b'athens'])
    # exit()
    # The eval feeds three vectors of word ids for a, b, c, each of
    # which is of size N, where N is the number of analogies we want to
    # evaluate in one batch.
    analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
    analogy_c = tf.placeholder(dtype=tf.int32)  # [N]
    # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
    # They all have the shape [N, emb_dim]
    a_emb = tf.gather(normalized_embeddings, analogy_a)  # a's embs
    b_emb = tf.gather(normalized_embeddings, analogy_b)  # b's embs
    c_emb = tf.gather(normalized_embeddings, analogy_c)  # c's embs
    # We expect that d's embedding vectors on the unit hyper-sphere is
    # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
    #   Bangkok Thailand Tokyo Japan .. Thailand - Bangkok = Japan - Tokyo
    target = c_emb + (b_emb - a_emb)
    # Compute cosine distance between each pair of target and vocab.
    # dist has shape [N, vocab_size].
    dist = tf.matmul(target, normalized_embeddings, transpose_b=True)
    # For each question (row in dist), find the top 4 words.
    _, pred_idx = tf.nn.top_k(dist, 4)
    def predict(analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = sess.run([pred_idx], {
            analogy_a: analogy[:, 0],
            analogy_b: analogy[:, 1],
            analogy_c: analogy[:, 2]
        })
        return idx

    # Evaluate analogy questions and reports accuracy.
    #  i.e. How many questions we get right at precision@1.
    correct = 0
    total = analogy_questions.shape[0]
    start = 0
    while start < total:
        limit = start + 1#2500
        sub = analogy_questions[start:limit, :] # question
        idx = predict(sub)      # 4 answers for every question
        # print('question:', tl.files.word_ids_to_words(sub[0], reverse_dictionary))
        # print('answers:', tl.files.word_ids_to_words(idx[0], reverse_dictionary))
        start = limit
        for question in xrange(sub.shape[0]):
            for j in xrange(4):
                # if one of the top 4 answers in correct, win !
                if idx[question, j] == sub[question, 3]:
                    # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                    print('!'*100)
                    correct += 1
                    break
                elif idx[question, j] in sub[question, :3]:
                    # We need to skip words already in the question.
                    continue
                else:
                    # The correct label is not the precision@1
                    break
    print()
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                             correct * 100.0 / total))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    main_word2vec_basic()













#
