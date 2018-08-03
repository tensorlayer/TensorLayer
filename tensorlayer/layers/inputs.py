#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import force_return_self

__all__ = [
    'InputLayer',
    'OneHotInputLayer',
    'Word2vecEmbeddingInputlayer',
    'EmbeddingInputlayer',
    'AverageEmbeddingInputlayer',
]


class InputLayer(Layer):
    """
    The :class:`InputLayer` class is the starting layer of a neural network.

    Parameters
    ----------
    inputs : placeholder or tensor
        The input of a network.
    name : str
        A unique layer name.

    """

    def __init__(self, inputs=None, name='input'):

        self.prev_layer = inputs
        self.name = name

        super(InputLayer, self).__init__()

    def __str__(self):

        additional_str = []

        try:
            additional_str.append("input shape: %s" % self.inputs.shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(InputLayer, self).__call__(prev_layer)

        self.outputs = self.inputs
        self._add_layers(self.outputs)


class OneHotInputLayer(Layer):
    """
    The :class:`OneHotInputLayer` class is the starting layer of a neural network, see ``tf.one_hot``.

    Parameters
    ----------
    inputs : placeholder or tensor
        The input of a network.
    depth : None or int
        If the input indices is rank N, the output will have rank N+1. The new axis is created at dimension `axis` (default: the new axis is appended at the end).
    on_value : None or number
        The value to represnt `ON`. If None, it will default to the value 1.
    off_value : None or number
        The value to represnt `OFF`. If None, it will default to the value 0.
    axis : None or int
        The axis.
    dtype : None or TensorFlow dtype
        The data type, None means tf.float32.
    name : str
        A unique layer name.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.int32, shape=[None])
    >>> net = tl.layers.OneHotInputLayer(x, depth=8, name='one_hot_encoding')
    (?, 8)

    """

    def __init__(self, inputs=None, depth=None, on_value=None, off_value=None, axis=None, dtype=None, name='input'):

        if depth is None:
            _err = "%s: depth  cannot be set to `None`. It leads to an undefined number of output units" % self.__class__.__name__
            raise RuntimeError(_err)

        self.prev_layer = inputs
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype
        self.name = name

        super(OneHotInputLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("input_shape: %s" % self.inputs.shape)
        except AttributeError:
            pass

        try:
            additional_str.append("output_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        self._parse_inputs(prev_layer)

        self.outputs = tf.one_hot(
            self.inputs, self.depth, on_value=self.on_value, off_value=self.off_value, axis=self.axis, dtype=self.dtype
        )
        self.out_shape = self.outputs.shape

        super(OneHotInputLayer, self).__call__(prev_layer)

        self._add_layers(self.outputs)


class Word2vecEmbeddingInputlayer(Layer):
    """
    The :class:`Word2vecEmbeddingInputlayer` class is a fully connected layer.
    For Word Embedding, words are input as integer index.
    The output is the embedded word vector.

    Parameters
    ----------
    inputs : placeholder or tensor
        The input of a network. For word inputs, please use integer index format, 2D tensor : [batch_size, num_steps(num_words)]
    train_labels : placeholder
        For word labels. integer index format
    vocabulary_size : int
        The size of vocabulary, number of words
    embedding_size : int
        The number of embedding dimensions
    num_sampled : int
        The mumber of negative examples for NCE loss
    nce_loss_args : dictionary
        The arguments for tf.nn.nce_loss()
    E_init : initializer
        The initializer for initializing the embedding matrix
    E_init_args : dictionary
        The arguments for embedding initializer
    nce_W_init : initializer
        The initializer for initializing the nce decoder weight matrix
    nce_W_init_args : dictionary
        The arguments for initializing the nce decoder weight matrix
    nce_b_init : initializer
        The initializer for initializing of the nce decoder bias vector
    nce_b_init_args : dictionary
        The arguments for initializing the nce decoder bias vector
    name : str
        A unique layer name

    Attributes
    ----------
    nce_cost : Tensor
        The NCE loss.
    outputs : Tensor
        The embedding layer outputs.
    normalized_embeddings : Tensor
        Normalized embedding matrix.

    Examples
    --------
    With TensorLayer : see ``tensorlayer/example/tutorial_word2vec_basic.py``

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 8
    >>> train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
    >>> train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
    >>> net = tl.layers.Word2vecEmbeddingInputlayer(inputs=train_inputs,
    ...     train_labels=train_labels, vocabulary_size=1000, embedding_size=200,
    ...     num_sampled=64, name='word2vec')
    (8, 200)
    >>> cost = net.nce_cost
    >>> train_params = net.all_params
    >>> cost = net.nce_cost
    >>> train_params = net.all_params
    >>> train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=train_params)
    >>> normalized_embeddings = net.normalized_embeddings

    Without TensorLayer : see ``tensorflow/examples/tutorials/word2vec/word2vec_basic.py``

    >>> train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
    >>> train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
    >>> embeddings = tf.Variable(
    ...     tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    >>> embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    >>> nce_weights = tf.Variable(
    ...     tf.truncated_normal([vocabulary_size, embedding_size],
    ...                    stddev=1.0 / math.sqrt(embedding_size)))
    >>> nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    >>> cost = tf.reduce_mean(
    ...    tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
    ...               inputs=embed, labels=train_labels,
    ...               num_sampled=num_sampled, num_classes=vocabulary_size,
    ...               num_true=1))

    References
    ----------
    `tensorflow/examples/tutorials/word2vec/word2vec_basic.py <https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>`__

    """

    def __init__(
            self,
            inputs=None,
            train_labels=None,
            vocabulary_size=80000,
            embedding_size=200,
            num_sampled=64,
            nce_loss_args=None,
            E_init=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
            E_init_args=None,
            nce_W_init=tf.truncated_normal_initializer(stddev=0.03),
            nce_W_init_args=None,
            nce_b_init=tf.constant_initializer(value=0.0),
            nce_b_init_args=None,
            dtype=None,
            name='word2vec',
    ):
        self.prev_layer = inputs
        self.train_labels = train_labels
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.E_init = E_init
        self.nce_W_init = nce_W_init
        self.nce_b_init = nce_b_init
        self.dtype = dtype
        self.name = name

        super(Word2vecEmbeddingInputlayer, self).__init__(
            nce_W_init_args=nce_W_init_args, nce_b_init_args=nce_b_init_args, nce_loss_args=nce_loss_args,
            E_init_args=E_init_args
        )

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embedding shape: %s" % self._emb_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        self._parse_inputs(prev_layer)

        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.

        self._emb_shape = [self.vocabulary_size, self.embedding_size]

        with tf.variable_scope(self.name):

            embeddings = self._get_tf_variable(
                name='embeddings', shape=self._emb_shape, dtype=self.dtype, initializer=self.E_init, **self.E_init_args
            )

            self.outputs = tf.nn.embedding_lookup(embeddings, self.inputs)
            self.out_shape = self.outputs.shape

            super(Word2vecEmbeddingInputlayer, self).__call__(prev_layer)

            # Construct the variables for the NCE loss (i.e. negative sampling)
            nce_weights = self._get_tf_variable(
                name='nce_weights', shape=self._emb_shape, dtype=self.dtype, initializer=self.nce_W_init,
                **self.nce_W_init_args
            )

            nce_biases = self._get_tf_variable(
                name='nce_biases', shape=(self.vocabulary_size, ), dtype=self.dtype, initializer=self.nce_b_init,
                **self.nce_b_init_args
            )

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels
            # each time we evaluate the loss.

            self.nce_cost = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights, biases=nce_biases, inputs=self.outputs, labels=self.train_labels,
                    num_sampled=self.num_sampled, num_classes=self.vocabulary_size, **self.nce_loss_args
                )
            )

            self.normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


class EmbeddingInputlayer(Layer):
    """
    The :class:`EmbeddingInputlayer` class is a look-up table for word embedding.

    Word content are accessed using integer indexes, then the output is the embedded word vector.
    To train a word embedding matrix, you can used :class:`Word2vecEmbeddingInputlayer`.
    If you have a pre-trained matrix, you can assign the parameters into it.

    Parameters
    ----------
    inputs : placeholder
        The input of a network. For word inputs.
        Please use integer index format, 2D tensor : (batch_size, num_steps(num_words)).
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    E_init : initializer
        The initializer for the embedding matrix.
    E_init_args : dictionary
        The arguments for embedding matrix initializer.
    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : tensor
        The embedding layer output is a 3D tensor in the shape: (batch_size, num_steps(num_words), embedding_size).

    Examples
    --------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 8
    >>> x = tf.placeholder(tf.int32, shape=(batch_size, ))
    >>> net = tl.layers.EmbeddingInputlayer(inputs=x, vocabulary_size=1000, embedding_size=50, dtype=tf.float32, name='embed')
    (8, 50)
    """

    def __init__(
            self,
            inputs=None,
            vocabulary_size=80000,
            embedding_size=200,
            E_init=tf.random_uniform_initializer(-0.1, 0.1),
            E_init_args=None,
            dtype=None,
            name='embedding',
    ):
        self.prev_layer = inputs
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.E_init = E_init
        self.dtype = dtype
        self.name = name

        super(EmbeddingInputlayer, self).__init__(E_init_args=E_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embedding shape: %s" % self._emb_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        self._parse_inputs(prev_layer)

        self._emb_shape = [self.vocabulary_size, self.embedding_size]

        with tf.variable_scope(self.name):

            embeddings = self._get_tf_variable(
                name='embeddings', shape=self._emb_shape, initializer=self.E_init, dtype=self.dtype, **self.E_init_args
            )

            self.outputs = tf.nn.embedding_lookup(embeddings, self.inputs)
            self.out_shape = self.outputs.shape

        super(EmbeddingInputlayer, self).__call__(prev_layer)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


class AverageEmbeddingInputlayer(Layer):
    """The :class:`AverageEmbeddingInputlayer` averages over embeddings of inputs.
    This is often used as the input layer for models like DAN[1] and FastText[2].

    Parameters
    ----------
    inputs : placeholder or tensor
        The network input.
        For word inputs, please use integer index format, 2D tensor: (batch_size, num_steps(num_words)).
    vocabulary_size : int
        The size of vocabulary.
    embedding_size : int
        The dimension of the embedding vectors.
    pad_value : int
        The scalar padding value used in inputs, 0 as default.
    embeddings_initializer : initializer
        The initializer of the embedding matrix.
    embeddings_kwargs : None or dictionary
        The arguments to get embedding matrix variable.
    name : str
        A unique layer name.

    References
    ----------
    - [1] Iyyer, M., Manjunatha, V., Boyd-Graber, J., & Daum’e III, H. (2015). Deep Unordered Composition Rivals Syntactic Methods for Text Classification. In Association for Computational Linguistics.
    - [2] Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). `Bag of Tricks for Efficient Text Classification. <http://arxiv.org/abs/1607.01759>`__

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 8
    >>> length = 5
    >>> x = tf.placeholder(tf.int32, shape=(batch_size, length))
    >>> net = tl.layers.AverageEmbeddingInputlayer(x, vocabulary_size=1000, embedding_size=50, name='avg')
    (8, 50)

    """

    def __init__(
            self,
            inputs,
            vocabulary_size,
            embedding_size,
            pad_value=0,
            embeddings_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            embeddings_kwargs=None,
            dtype=None,
            name='average_embedding',
    ):
        self.prev_layer = inputs
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pad_value = pad_value
        self.embeddings_initializer = embeddings_initializer
        self.dtype = dtype
        self.name = name

        super(AverageEmbeddingInputlayer, self).__init__(embeddings_kwargs=embeddings_kwargs)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embedding shape: %s" % self._emb_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        if prev_layer.get_shape().ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        self._parse_inputs(prev_layer)

        self._emb_shape = [self.vocabulary_size, self.embedding_size]

        with tf.variable_scope(self.name):

            embeddings = self._get_tf_variable(
                name='embeddings', shape=self._emb_shape, dtype=self.dtype, initializer=self.embeddings_initializer,
                **self.embeddings_kwargs
            )

            word_embeddings = tf.nn.embedding_lookup(
                embeddings,
                self.inputs,
                name='word_embeddings',
            )
            # Zero out embeddings of pad value
            masks = tf.not_equal(self.inputs, self.pad_value, name='masks')

            word_embeddings *= tf.cast(tf.expand_dims(masks, axis=-1), dtype=self.dtype)

            sum_word_embeddings = tf.reduce_sum(word_embeddings, axis=1)

            # Count number of non-padding words in each sentence
            sentence_lengths = tf.count_nonzero(
                masks,
                axis=1,
                keepdims=True,
                dtype=self.dtype,
                name='sentence_lengths',
            )

            sentence_embeddings = tf.divide(
                sum_word_embeddings,
                sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
                name='sentence_embeddings'
            )

            self.out_shape = sentence_embeddings.shape

        super(AverageEmbeddingInputlayer, self).__call__(prev_layer)

        self.outputs = sentence_embeddings

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)
