#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

__all__ = [
    'OneHot',
    'Word2vecEmbedding',
    'Embedding',
    'AverageEmbedding',
]

class OneHot(Layer):
    """
    The :class:`OneHot` class is the starting layer of a neural network, see ``tf.one_hot``.

    Parameters
    ----------
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
    >>> net = tl.layers.OneHot(x, depth=8)
    (?, 8)

    """

    def __init__(self, depth=None, on_value=None, off_value=None, axis=None, dtype=None, name=None):  #'input'):

        # super(OneHotInput, self).__init__(prev_layer=inputs, name=name)
        super().__init__(name)
        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.dtype = dtype
        logging.info("OneHotInput  %s: %s" % (self.name, str(inputs.shape.as_list())))

        if self.depth is None:
            raise RuntimeError(self.__class__.__name__ + ": depth == None the number of output units is undefined")

    def build(self, inputs_shape):
        pass

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network.
        """
        outputs = tf.one_hot(
            inputs, self.depth, on_value=self.on_value, off_value=self.off_value, axis=self.axis, dtype=self.dtype
        )
        return outputs


class Word2vecEmbedding(Layer):
    """
    The :class:`Word2vecEmbedding` class is a fully connected layer.
    For Word Embedding, words are input as integer index.
    The output is the embedded word vector.

    Parameters
    ----------
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
    nce_W_init : initializer
        The initializer for initializing the nce decoder weight matrix
    nce_b_init : initializer
        The initializer for initializing of the nce decoder bias vector
    name : str
        A unique layer name

    Attributes
    ----------
    outputs : Tensor
        The embedding layer outputs.
    normalized_embeddings : Tensor
        Normalized embedding matrix.
    nce_weights : Tensor
        The NCE weights
    nce_biases: Tensor
        The NCE biases

    Examples
    --------
    Word2Vec With TensorLayer (Example in `examples/text_word_embedding/tutorial_word2vec_basic.py`)

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> batch_size = 8
    >>> train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    >>> train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    >>> net_in = tl.layers.Input([batch_size], dtype=tf.int32)
    >>> emb_net = tl.layers.Word2vecEmbedding(
    >>>       vocabulary_size=vocabulary_size,
    >>>       embedding_size=embedding_size,
    >>>       num_sampled=num_sampled,
    >>>       nce_loss_args={},
    >>>       E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
    >>>       nce_W_init=tl.initializers.truncated_normal(stddev=float(1.0 / np.sqrt(embedding_size))),
    >>>       nce_b_init=tl.initializers.constant(value=0.0),
    >>>       name='word2vec_layer',
    >>> )(net_in)

    >>> model = tl.models.Model(inputs=net_in, outputs=emb_net, name="word2vec_model")

    >>> nce_cost = tf.reduce_mean(
    >>>     input_tensor=tf.nn.nce_loss(
    >>>         weights=emb_net.nce_weights,
    >>>         biases=emb_net.nce_biases,
    >>>         inputs=model(train_inputs, is_train=True),
    >>>         labels=train_labels,  #self.train_labels,
    >>>         num_sampled=emb_net.num_sampled,
    >>>         num_classes=emb_net.vocabulary_size,
    >>>         **emb_net.nce_loss_args
    >>>    )
    >>> )

    >>> train_params = model.weights
    >>> train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(nce_cost, var_list=train_params)

    >>> normalized_embeddings = emb_net.normalized_embeddings

    References
    ----------
    `tensorflow/examples/tutorials/word2vec/word2vec_basic.py <https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>`__

    """

    def __init__(
            self,
            vocabulary_size=80000,
            embedding_size=200,
            num_sampled=64,
            nce_loss_args=None,
            E_init=tl.initializers.random_uniform(minval=-1.0, maxval=1.0),
            nce_W_init=tl.initializers.truncated_normal(stddev=0.03),
            nce_b_init=tl.initializers.constant(value=0.0),
            name=None,  #'word2vec',
    ):

        super(Word2vecEmbedding, self).__init__(name)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.nce_loss_args = nce_loss_args
        self.E_init = E_init
        self.nce_W_init = nce_W_init
        self.nce_b_init = nce_b_init
        logging.info("Word2vecEmbedding %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor
        """
        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transferring a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.

        self.embeddings = self._get_weights(
            "embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init,
        )

        self.normalized_embeddings = tf.nn.l2_normalize(self.embeddings, 1)

        # Construct the variables for the NCE loss (i.e. negative sampling)
        self.nce_weights = self._get_weights(
            "nce_weights", shape=(self.vocabulary_size, self.embedding_size), init=self.nce_W_init,
        )

        self.nce_biases = self._get_weights(
            "nce_biases", shape=(self.vocabulary_size,), init=self.nce_b_init,
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network
        """
        outputs = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs)

        return outputs


class Embedding(Layer):
    """
    The :class:`Embedding` class is a look-up table for word embedding.

    Word content are accessed using integer indexes, then the output is the embedded word vector.
    To train a word embedding matrix, you can used :class:`Word2vecEmbedding`.
    If you have a pre-trained matrix, you can assign the parameters into it.

    Parameters
    ----------
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
    >>> net = tl.layers.Input([batch_size, 100], dtype=tf.int32)
    >>> net = tl.layers.Embedding(vocabulary_size=1000, embedding_size=50, name='embed')(net)
    >>> print(net.outputs)
    <tf.Tensor 'embedding_lookup/Identity:0' shape=(8, 100, 50) dtype=float32>

    """

    def __init__(
            self,
            vocabulary_size=80000,
            embedding_size=200,
            E_init=tl.initializers.random_uniform(-0.1, 0.1),
            name=None,  #'embedding',
    ):
        super(Embedding, self).__init__(name)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.E_init = E_init

        logging.info("Embedding %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor
        """

        self.embeddings = self._get_weights(
            "embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init,
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Tensor
            The input of a network.
        """
        outputs = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs)
        return outputs


class AverageEmbedding(Layer):
    """The :class:`AverageEmbedding` averages over embeddings of inputs.
    This is often used as the input layer for models like DAN[1] and FastText[2].

    Parameters
    ----------
    vocabulary_size : int
        The size of vocabulary.
    embedding_size : int
        The dimension of the embedding vectors.
    pad_value : int
        The scalar padding value used in inputs, 0 as default.
    E_init : initializer
        The initializer of the embedding matrix.
    name : str
        A unique layer name.

    Attributes
    ----------
    outputs : tensor
        The embedding layer output is a 2D tensor in the shape: (batch_size, embedding_size).

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
    >>> net = tl.layers.Input([batch_size, length], dtype=tf.int32)
    >>> net = tl.layers.AverageEmbedding(vocabulary_size=1000, embedding_size=50, name='avg')(net)
    >>> print(net.outputs)
    <tf.Tensor 'sentence_embeddings:0' shape=(8, 50) dtype=float32>

    """

    def __init__(
            self,
            vocabulary_size,
            embedding_size,
            pad_value=0,
            E_init=tl.initializers.random_uniform(-0.1, 0.1),
            name=None,  # 'average_embedding',
    ):

        super(AverageEmbedding, self).__init__(name)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pad_value = pad_value
        self.E_init = E_init
        logging.info("AverageEmbedding %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

    def build(self, inputs_shape):
        """
        Parameters
        ----------
        inputs_shape : tuple
            the shape of inputs tensor.
        """
        if len(inputs_shape) != 2:
            raise ValueError('inputs must be of size (batch_size, sentence_length)')

        self.embeddings = self._get_weights(
            "embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init,
        )

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : tensor
            The network input.
            For word inputs, please use integer index format, 2D tensor: (batch_size, sentence_length).
        """
        word_embeddings = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=inputs,
            name='word_embeddings',
        )

        # Zero out embeddings of pad value
        masks = tf.not_equal(inputs, self.pad_value, name='masks')
        word_embeddings *= tf.cast(
            tf.expand_dims(masks, axis=-1),
            dtype=tf.float32
        )
        sum_word_embeddings = tf.reduce_sum(input_tensor=word_embeddings, axis=1)

        # Count number of non-padding words in each sentence
        sentence_lengths = tf.math.count_nonzero(
            masks,
            axis=1,
            keepdims=True,
            dtype=tf.float32,
            name='sentence_lengths',
        )

        sentence_embeddings = tf.divide(
            sum_word_embeddings,
            sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
            name='sentence_embeddings'
        )

        outputs = sentence_embeddings

        return outputs
