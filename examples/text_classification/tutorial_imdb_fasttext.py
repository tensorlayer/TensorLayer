#!/usr/bin/env python
"""
This demo implements FastText[1] for sentence classification. This demo should be run in eager mode and
can be slower than the corresponding demo in graph mode.

FastText is a simple model for text classification with performance often close
to state-of-the-art, and is useful as a solid baseline.

There are some important differences between this implementation and what
is described in the paper. Instead of Hogwild! SGD[2], we use Adam optimizer
with mini-batches. Hierarchical softmax is also not supported; if you have
a large label space, consider utilizing candidate sampling methods provided
by TensorFlow[3].

After 5 epochs, you should get test accuracy around 90.3%.

[1] Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016).
    Bag of Tricks for Efficient Text Classification.
    http://arxiv.org/abs/1607.01759

[2] Recht, B., Re, C., Wright, S., & Niu, F. (2011).
    Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    In Advances in Neural Information Processing Systems 24 (pp. 693–701).

[3] https://www.tensorflow.org/api_guides/python/nn#Candidate_Sampling

"""
import array
import hashlib
import os
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.models import *

# Hashed n-grams with 1 < n <= N_GRAM are included as features
# in addition to unigrams.
N_GRAM = 2

# Size of vocabulary; less frequent words will be treated as "unknown"
VOCAB_SIZE = 100000

# Number of buckets used for hashing n-grams
N_BUCKETS = 1000000

# Size of the embedding vectors
EMBEDDING_SIZE = 50

# Number of epochs for which the model is trained
N_EPOCH = 5

# Number of steps for printing
N_STEPS_TO_PRINT = 100

# Size of training mini-batches
BATCH_SIZE = 32

# Learning rate
LEARNING_RATE = 0.01

# Path to which to save the trained model
MODEL_FILE_PATH = 'model_dynamic.hdf5'


class FastTextModel(Model):
    """  Model structure and forwarding of FastText """

    def __init__(self, vocab_size, embedding_size, n_labels, name='fasttext'):
        super(FastTextModel, self).__init__(name=name)

        self.avg_embed = AverageEmbedding(vocab_size, embedding_size)
        self.dense1 = Dense(n_units=10, in_channels=embedding_size)
        self.dense2 = Dense(n_units=n_labels, in_channels=10)

    def forward(self, x):
        z = self.avg_embed(x)
        z = self.dense1(z)
        z = self.dense2(z)
        return z


def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    """Augment unigram features with hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return unigrams + [hash_ngram(ngram) for i in range(2, n + 1) for ngram in get_ngrams(i)]


def load_and_preprocess_imdb_data(n_gram=None):
    """Load IMDb data and augment with hashed n-gram features."""
    tl.logging.info("Loading and preprocessing IMDB data.")

    X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(nb_words=VOCAB_SIZE)

    if n_gram is not None:
        X_train = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_train])
        X_test = np.array([augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram) for x in X_test])

    return X_train, y_train, X_test, y_test


def train_test_and_save_model():
    X_train, y_train, X_test, y_test = load_and_preprocess_imdb_data(N_GRAM)
    model = FastTextModel(
        vocab_size=VOCAB_SIZE + N_BUCKETS,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    if os.path.exists(MODEL_FILE_PATH):
        # loading pre-trained model if applicable
        model.load_weights(MODEL_FILE_PATH)
    else:
        # training
        model.train()

        for epoch in range(N_EPOCH):
            start_time = time.time()
            print('Epoch %d/%d' % (epoch + 1, N_EPOCH))
            train_accuracy = list()
            for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):

                # forward and define the loss function
                # TODO: use tf.function to speed up
                with tf.GradientTape() as tape:
                    y_pred = model(tl.prepro.pad_sequences(X_batch))
                    cost = tl.cost.cross_entropy(y_pred, y_batch, name='cost')

                # backward, calculate gradients and update the weights
                grad = tape.gradient(cost, model.trainable_weights)
                optimizer.apply_gradients(zip(grad, model.trainable_weights))

                # calculate the accuracy
                predictions = tf.argmax(y_pred, axis=1, output_type=tf.int32)
                are_predictions_correct = tf.equal(predictions, y_batch)
                accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))

                train_accuracy.append(accuracy)
                if len(train_accuracy) % N_STEPS_TO_PRINT == 0:
                    print(
                        "\t[%d/%d][%d]accuracy " % (epoch + 1, N_EPOCH, len(train_accuracy)),
                        np.mean(train_accuracy[-N_STEPS_TO_PRINT:])
                    )

            print("\tSummary: time %.5fs, overall accuracy" % (time.time() - start_time), np.mean(train_accuracy))

    # evaluation and testing
    model.eval()

    # forward and calculate the accuracy
    y_pred = model(tl.prepro.pad_sequences(X_test))
    predictions = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    are_predictions_correct = tf.equal(predictions, y_test)
    test_accuracy = tf.reduce_mean(tf.cast(are_predictions_correct, tf.float32))

    print('Test accuracy: %.5f' % test_accuracy)

    # saving the model
    model.save_weights(MODEL_FILE_PATH)


if __name__ == '__main__':
    train_test_and_save_model()
