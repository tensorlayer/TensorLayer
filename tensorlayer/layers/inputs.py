#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

__all__ = [
    'Input',
    'OneHotInput',              # --> OneHot
    'Word2vecEmbeddingInput',   # --> Word2vecEmbedding
    'EmbeddingInput',           # --> Embedding
    'AverageEmbeddingInput',    # --> AverageEmbeddin
]


class Input(Layer):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    name : None or str
        A unique layer name.

    """

    def __init__(self, shape, name=None):  #'input'):
        # super(InputLayer, self).__init__(prev_layer=inputs, name=name)
        super().__init__(name)

        logging.info("Input  %s: %s" % (self.name, str(shape)))

        shape_without_none = [_ if _ is not None else 1 for _ in shape]
        self.outputs = self.forward(
            tf.initializers.constant(value=0.0)(shape_without_none),
            is_train=False
        )

    def __call__(self, prev_layer):
        # FIXME: better exception raising
        raise Exception(
            "__call__() of Input deactivated"
        )

    def build(self, inputs_shape):
        # FIXME: documentation need double check
        """
        no weights to define
        """
        pass

    def forward(self, inputs, is_train):
        # FIXME: documentation need double check
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network.
        is_train: bool
            train (True) or test (False)
        """
        return inputs


class OneHotInput(Layer):
    """
    The :class:`OneHotInput` class is the starting layer of a neural network, see ``tf.one_hot``.

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
    >>> net = tl.layers.OneHotInput(x, depth=8)
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

    def build(self, inputs):
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


class Word2vecEmbeddingInput(Layer):
    """
    The :class:`Word2vecEmbeddingInput` class is a fully connected layer.
    For Word Embedding, words are input as integer index.
    The output is the embedded word vector.

    Parameters
    ----------
    # train_labels : placeholder
    #     For word labels. integer index format
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
    >>> net = tl.layers.Word2vecEmbeddingInput(inputs=train_inputs,
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
            # inputs,
            # train_labels=None,
            vocabulary_size=80000,
            embedding_size=200,
            num_sampled=64,
            nce_loss_args=None,
            E_init=tf.compat.v1.initializers.random_uniform(minval=-1.0, maxval=1.0),
            E_init_args=None,
            nce_W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.03),
            nce_W_init_args=None,
            nce_b_init=tf.compat.v1.initializers.constant(value=0.0),
            nce_b_init_args=None,
            name=None,  #'word2vec',
    ):

        # super(Word2vecEmbeddingInput, self).__init__(
        #     prev_layer=inputs, nce_loss_args=nce_loss_args, E_init_args=E_init_args, nce_W_init_args=nce_W_init_args,
        #     nce_b_init_args=nce_b_init_args, name=name
        # )
        super().__init__(name)
        # self.train_labels = train_labels
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.nce_loss_args = nce_loss_args
        self.E_init = E_init
        self.E_init_args = E_init_args
        self.nce_W_init = nce_W_init
        self.nce_W_init_args = nce_W_init_args
        self.nce_b_init = nce_b_init
        self.nce_b_init_args = nce_b_init_args
        logging.info("Word2vecEmbeddingInput %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

    def build(self, inputs):
        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.

        # self.embeddings = tf.compat.v1.get_variable(
        #     name=self.name + '/embeddings', shape=(self.vocabulary_size, self.embedding_size), initializer=self.E_init,
        #     dtype=LayersConfig.tf_dtype, **self.E_init_args
        # )
        self.embeddings = self._get_weights("embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init, init_args=self.E_init_args)

        self.normalized_embeddings = tf.nn.l2_normalize(self.embeddings, 1)

        # Construct the variables for the NCE loss (i.e. negative sampling)
        # self.nce_weights = tf.compat.v1.get_variable(
        #     name=self.name + '/nce_weights', shape=(self.vocabulary_size, self.embedding_size),
        #     initializer=self.nce_W_init, dtype=LayersConfig.tf_dtype, **self.nce_W_init_args
        # )
        self.nce_weights = self._get_weights("nce_weights", shape=(self.vocabulary_size, self.embedding_size), init=self.nce_W_init, init_args=self.nce_W_init_args)

        # self.nce_biases = tf.compat.v1.get_variable(
        #     name=self.name + '/nce_biases', shape=(self.vocabulary_size), initializer=self.nce_b_init,
        #     dtype=LayersConfig.tf_dtype, **self.nce_b_init_args
        # )
        self.nce_biases = self._get_weights("nce_biases", shape=(self.vocabulary_size), init=self.nce_b_init, init_args=self.nce_b_init_args)

    def forward(self, inputs, train_labels):
        """
        Parameters
        ----------
        inputs : input tensor
            The input of a network. For word inputs, please use integer index format, 2D tensor : [batch_size, num_steps(num_words)]
        train_labels : input tensor
            For word labels. integer index format
        """
        outputs = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs)

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels
        # each time we evaluate the loss.
        self.nce_cost = tf.reduce_mean(
            input_tensor=tf.nn.nce_loss(
                weights=self.nce_weights, biases=self.nce_biases, inputs=outputs, labels=train_labels, #self.train_labels,
                num_sampled=self.num_sampled, num_classes=self.vocabulary_size, **self.nce_loss_args
            )
        )
        return outputs


class EmbeddingInput(Layer):
    """
    The :class:`EmbeddingInput` class is a look-up table for word embedding.

    Word content are accessed using integer indexes, then the output is the embedded word vector.
    To train a word embedding matrix, you can used :class:`Word2vecEmbeddingInput`.
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
    >>> x = tf.placeholder(tf.int32, shape=(batch_size, ))
    >>> net = tl.layers.EmbeddingInput(inputs=x, vocabulary_size=1000, embedding_size=50, name='embed')
    (8, 50)

    """

    def __init__(
            self,
            # inputs,
            vocabulary_size=80000,
            embedding_size=200,
            E_init=tf.compat.v1.initializers.random_uniform(-0.1, 0.1),
            E_init_args=None,
            name=None,  #'embedding',
    ):
        # super(EmbeddingInput, self).__init__(prev_layer=inputs, E_init_args=E_init_args, name=name)
        super().__init__(name)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.E_init = E_init
        self.E_init_args = E_init_args

        logging.info("EmbeddingInput %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

    def build(self, inputs):
        # self.embeddings = tf.compat.v1.get_variable(
        #     name=self.name + '/embeddings', shape=(self.vocabulary_size, self.embedding_size), initializer=self.E_init,
        #     dtype=LayersConfig.tf_dtype, **self.E_init_args
        # )
        self.embeddings = self._get_weights("embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init, init_args=self.E_init_args)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : placeholder
            The input of a network. For word inputs.
            Please use integer index format, 2D tensor : (batch_size, num_steps(num_words)).
        """
        outputs = tf.nn.embedding_lookup(params=self.embeddings, ids=inputs)
        return outputs


class AverageEmbeddingInput(Layer):
    """The :class:`AverageEmbeddingInput` averages over embeddings of inputs.
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
    E_init_args : None or dictionary
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
    >>> net = tl.layers.AverageEmbeddingInput(x, vocabulary_size=1000, embedding_size=50, name='avg')
    (8, 50)

    """

    def __init__(
            self,
            # inputs,
            vocabulary_size,
            embedding_size,
            pad_value=0,
            E_init=tf.compat.v1.initializers.random_uniform(-0.1, 0.1),
            # E_init=None,
            name=None,  # 'average_embedding',
    ):

        # super(AverageEmbeddingInput, self).__init__(prev_layer=inputs, embeddings_kwargs=embeddings_kwargs, name=name)
        super().__init__(name)
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pad_value = pad_value
        self.E_init = E_init
        self.E_init_args = E_init_args
        logging.info("AverageEmbeddingInput %s: (%d, %d)" % (self.name, self.vocabulary_size, self.embedding_size))

        # if embeddings_kwargs is None:
        #     embeddings_kwargs = {}
    def build(self, inputs):
        if inputs.shape.ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        # self.embeddings = tf.compat.v1.get_variable(
        #     name=self.name + '/embeddings', shape=(self.vocabulary_size, self.embedding_size),
        #     initializer=self.E_init, dtype=LayersConfig.tf_dtype, **self.E_init_argst
        # )
        self.embeddings = self._get_weights("embeddings", shape=(self.vocabulary_size, self.embedding_size), init=self.E_init, init_args=self.E_init_args)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : input tensor
            The network input.
            For word inputs, please use integer index format, 2D tensor: (batch_size, num_steps(num_words)).
        """
        word_embeddings = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=inputs,
            name='word_embeddings',
        )
        # Zero out embeddings of pad value
        masks = tf.not_equal(inputs, pad_value, name='masks')
        word_embeddings *= tf.cast(
            tf.expand_dims(masks, axis=-1),
            # dtype=LayersConfig.tf_dtype,
        )
        sum_word_embeddings = tf.reduce_sum(input_tensor=word_embeddings, axis=1)

        # Count number of non-padding words in each sentence
        sentence_lengths = tf.compat.v1.count_nonzero(
            masks,
            axis=1,
            keepdims=True,
            # dtype=LayersConfig.tf_dtype,
            name='sentence_lengths',
        )

        sentence_embeddings = tf.divide(
            sum_word_embeddings,
            sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
            name='sentence_embeddings'
        )

        outputs = sentence_embeddings

        return outputs
