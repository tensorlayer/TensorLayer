#!/usr/bin/env python

__doc__ = """

This demo implements FastText[1] for sentence classification. FastText is a
simple model for text classification with performance often close to
state-of-the-art, and is useful as a solid baseline.

There are some important differences between this implementation and what
is described in the paper. Instead of Hogwild! SGD[2], we use Adam optimizer
with mini-batches. Hierarchical softmax is also not supported; if you have
a large label space, consider utilizing candidate sampling methods provided
by TensorFlow[3].

After 5 epochs, you should get test accuracy close to 90.9%.

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
import time

import tensorflow as tf
import tensorlayer as tl
import numpy as np


# Hashed n-grams with 1 < n <= N_GRAM are included as features
# in addition to unigrams.
N_GRAM = 2

# Size of vocabulary; less frequent works will be treated as "unknown"
VOCAB_SIZE = 100000

# Number of buckets used for hashing n-grams
N_BUCKETS = 1000000

# Size of the embedding vectors
EMBEDDING_SIZE = 50

# Number of epochs for which the model is trained
N_EPOCH = 5

# Size of training mini-batches
BATCH_SIZE = 32

# Path to which to save the trained model
MODEL_FILE_PATH = 'model.npz'


class FastTestEmbeddingInputLayer(tl.layers.Layer):
    def __init__(
            self, inputs, vocabulary_size, embedding_size,
            name='fasttext_layer',
            embeddings_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            embeddings_kwargs=None):
        """FastText Embedding input layer for sentences.

        :param inputs: input placeholder or tensor; zeros are paddings
        :param vocabulary_size: and integer, the size of vocabulary
        :param embedding_size: and integer, the dimension of embedding vectors
        :param name: a string, the name of the layer
        :param embeddings_initializer: the initializer of the embedding matrix
        :param embeddings_kwargs: kwargs to get embedding matrix variable
        """
        super().__init__(name=name)

        if inputs.get_shape().ndims != 2:
            raise ValueError(
                'inputs must be of size batch_size * batch_sentence_length')

        self.inputs = inputs

        print(f"  [TL] FastTestEmbeddingInputLayer {self.name}:"
              f" ({vocabulary_size}, {embedding_size})")

        with tf.variable_scope(name):
            self.embeddings = tf.get_variable(
                name='embeddings',
                shape=(vocabulary_size, embedding_size),
                initializer=embeddings_initializer,
                **(embeddings_kwargs or {}),
            )
            word_embeddings = tf.nn.embedding_lookup(
                self.embeddings, self.inputs,
                name='word_embeddings',
            )

            # Masks used to ignore padding words
            masks = tf.expand_dims(
                tf.sign(self.inputs),
                axis=-1,
                name='masks',
            )
            sum_word_embeddings = tf.reduce_sum(
                word_embeddings * tf.cast(masks, tf.float32),
                axis=1,
            )

            # Count number of non-padding words in each sentence
            # Used to commute average word embeddings in sentences
            sentence_lengths = tf.count_nonzero(
                self.inputs,
                axis=1,
                keep_dims=True,
                dtype=tf.float32,
                name='sentence_lengths',
            )

            sentence_embeddings = tf.divide(
                sum_word_embeddings,
                sentence_lengths,
                name='sentence_embeddings'
            )

        self.outputs = sentence_embeddings
        self.all_layers = [self.outputs]
        self.all_params = [self.embeddings]
        self.all_drop = {}


class FastTextClassifier(object):
    """Simple wrapper class for creating the graph of FastText classifier."""
    def __init__(self, vocab_size, embedding_size, n_labels):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_labels = n_labels

        self.inputs = tf.placeholder(
            tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(
            tf.int32, shape=[None], name='labels')

        # Network structure
        network = FastTestEmbeddingInputLayer(
            self.inputs, self.vocab_size, self.embedding_size)
        self.network = tl.layers.DenseLayer(network, self.n_labels)

        # Training operation
        cost = tl.cost.cross_entropy(
            self.network.outputs,
            self.labels,
            name='cost'
        )
        self.train_op = tf.train.AdamOptimizer().minimize(cost)

        # Predictions
        self.prediction_probs = tf.nn.softmax(self.network.outputs)
        self.predictions = tf.argmax(
            self.network.outputs, axis=1, output_type=tf.int32)

        # Evaluation
        are_predictions_correct = tf.equal(self.predictions, self.labels)
        self.accuracy = tf.reduce_mean(
            tf.cast(are_predictions_correct, tf.float32))

    def save(self, sess, filename):
        tl.files.save_npz(self.network.all_params, name=filename, sess=sess)

    def load(self, sess, filename):
        tl.files.load_and_assign_npz(sess, name=filename, network=self.network)


def augment_with_ngrams(unigrams, unigram_vocab_size, n_buckets, n=2):
    """Augment unigram features with hashed n-gram features."""
    def get_ngrams(n):
        return list(zip(*[
            unigrams[i:]
            for i in range(n)
        ]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return unigrams + [
        hash_ngram(ngram)
        for i in range(2, n + 1)
        for ngram in get_ngrams(i)
    ]


def load_and_preprocess_imdb_data(n_gram=None):
    """Load IMDb data and augment with hashed n-gram features."""
    X_train, y_train, X_test, y_test = \
        tl.files.load_imdb_dataset(nb_words=VOCAB_SIZE)

    if n_gram is not None:
        X_train = np.array([
            augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram)
            for x in X_train
        ])
        X_test = np.array([
            augment_with_ngrams(x, VOCAB_SIZE, N_BUCKETS, n=n_gram)
            for x in X_test
        ])

    return X_train, y_train, X_test, y_test


def train_test_and_save_model():
    X_train, y_train, X_test, y_test = load_and_preprocess_imdb_data(N_GRAM)
    classifier = FastTextClassifier(
        vocab_size=VOCAB_SIZE + N_BUCKETS,
        embedding_size=EMBEDDING_SIZE,
        n_labels=2,
    )

    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)

        for epoch in range(N_EPOCH):
            start_time = time.time()
            print(f'Epoch {epoch + 1}/{N_EPOCH}', end='')
            for X_batch, y_batch in tl.iterate.minibatches(
                    X_train, y_train, batch_size=BATCH_SIZE, shuffle=True):
                sess.run(classifier.train_op, feed_dict={
                    classifier.inputs: tl.prepro.pad_sequences(X_batch),
                    classifier.labels: y_batch,
                })

            print(f'\t{time.time() - start_time:.2f}s')

        test_accuracy = sess.run(classifier.accuracy, feed_dict={
            classifier.inputs: tl.prepro.pad_sequences(X_test),
            classifier.labels: y_test,
        })
        print(f'Test accuracy: {test_accuracy:.5f}')

        classifier.save(sess, MODEL_FILE_PATH)


if __name__ == '__main__':
    train_test_and_save_model()
