#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import deprecated_args

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
    name : str
        A unique layer name.

    """

    def __init__(self, name='input'):

        self.name = name

        super(InputLayer, self).__init__()

    def __str__(self):

        additional_str = []

        try:
            additional_str.append("input shape: %s" % self._temp_data['inputs'].shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = self._temp_data['inputs']


class OneHotInputLayer(Layer):
    """
    The :class:`OneHotInputLayer` class is the starting layer of a neural network, see ``tf.one_hot``.

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
    output_dtype : None or TensorFlow dtype
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

    def __init__(self, depth, on_value=None, off_value=None, axis=None, output_dtype=tf.float32, name='input'):

        if depth is None:
            _err = "%s: depth  cannot be set to `None`. It leads to an undefined number of output units" % self.__class__.__name__
            raise RuntimeError(_err)

        self.depth = depth
        self.on_value = on_value
        self.off_value = off_value
        self.axis = axis
        self.output_dtype = output_dtype
        self.name = name

        super(OneHotInputLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("input_shape: %s" % self._temp_data['inputs'].shape)
        except AttributeError:
            pass

        try:
            additional_str.append("output_shape: %s" % self._temp_data['outputs'].get_shape())
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        self._temp_data['outputs'] = tf.one_hot(
            self._temp_data['inputs'],
            self.depth,
            on_value=self.on_value,
            off_value=self.off_value,
            axis=self.axis,
            dtype=self.output_dtype
        )


class Word2vecEmbeddingInputlayer(Layer):
    """
    The :class:`Word2vecEmbeddingInputlayer` class is a fully connected layer.
    For Word Embedding, words are input as integer index.
    The output is the embedded word vector.

    Parameters
    ----------
    vocabulary_size : int
        The size of vocabulary, number of words
    embedding_size : int
        The number of embedding dimensions
    embeddings_dtype : TF Data Type (default: tf.float32)
        The dtype of the embeddings
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
    >>> train_params = net.all_weights
    >>> cost = net.nce_cost
    >>> train_params = net.all_weights
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
        vocabulary_size=80000,
        embedding_size=200,
        num_sampled=64,
        nce_loss_args=None,
        embeddings_dtype=tf.float32,
        E_init=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
        E_init_args=None,
        nce_W_init=tf.truncated_normal_initializer(stddev=0.03),
        nce_W_init_args=None,
        nce_b_init=tf.constant_initializer(value=0.0),
        nce_b_init_args=None,
        name='word2vec',
    ):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings_shape = [self.vocabulary_size, self.embedding_size]

        self.num_sampled = num_sampled
        self.E_init = E_init
        self.nce_W_init = nce_W_init
        self.nce_b_init = nce_b_init
        self.embeddings_dtype = embeddings_dtype
        self.name = name

        super(Word2vecEmbeddingInputlayer, self).__init__(
            nce_W_init_args=nce_W_init_args,
            nce_b_init_args=nce_b_init_args,
            nce_loss_args=nce_loss_args,
            E_init_args=E_init_args
        )

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embeddings shape: %s" % self.embeddings_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("embeddings dtype: %s" % self.embeddings_dtype)
        except AttributeError:
            pass

        return self._str(additional_str)

    def __call__(self, prev_layer, train_labels, is_train=True):
        """
        prev_layer : :class:`Layer`
            Previous layer.
        train_labels : placeholder
            For word labels. integer index format
        is_train: boolean (default: True)
            Set the TF Variable in training mode and may impact the behaviour of the layer.
        """
        return super(Word2vecEmbeddingInputlayer, self).__call__(
            prev_layer=[prev_layer, train_labels], is_train=is_train
        )

    def build(self):

        input_plh = self._temp_data['inputs'][0]
        train_labels_plh = self._temp_data['inputs'][1]

        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.

        with tf.variable_scope(self.name):

            embeddings = self._get_tf_variable(
                name='embeddings',
                shape=self.embeddings_shape,
                dtype=self.embeddings_dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.E_init,
                **self.E_init_args
            )

            self._temp_data['outputs'] = tf.nn.embedding_lookup(embeddings, input_plh)

            # Construct the variables for the NCE loss (i.e. negative sampling)
            nce_weights = self._get_tf_variable(
                name='nce_weights',
                shape=self.embeddings_shape,
                dtype=self.embeddings_dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.nce_W_init,
                **self.nce_W_init_args
            )

            nce_biases = self._get_tf_variable(
                name='nce_biases',
                shape=(self.vocabulary_size, ),
                dtype=self.embeddings_dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.nce_b_init,
                **self.nce_b_init_args
            )

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels
            # each time we evaluate the loss.

            self._temp_data['nce_cost'] = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    inputs=self._temp_data['outputs'],
                    labels=train_labels_plh,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size,
                    **self.nce_loss_args
                )
            )

            self._temp_data['normalized_embeddings'] = tf.nn.l2_normalize(embeddings, 1)


class EmbeddingInputlayer(Layer):
    """
    The :class:`EmbeddingInputlayer` class is a look-up table for word embedding.

    Word content are accessed using integer indexes, then the output is the embedded word vector.
    To train a word embedding matrix, you can used :class:`Word2vecEmbeddingInputlayer`.
    If you have a pre-trained matrix, you can assign the parameters into it.

    Parameters
    ----------
        Please use integer index format, 2D tensor : (batch_size, num_steps(num_words)).
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    embeddings_dtype : TF Data Type (default: tf.float32)
        The dtype of the embeddings
    E_init : initializer
        The initializer for the embedding matrix.
    E_init_args : dictionary
        The arguments for embedding matrix initializer.
    name : str
        A unique layer name.

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
        vocabulary_size=80000,
        embedding_size=200,
        embeddings_dtype=tf.float32,
        E_init=tf.random_uniform_initializer(-0.1, 0.1),
        E_init_args=None,
        name='embedding',
    ):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings_shape = [self.vocabulary_size, self.embedding_size]

        self.E_init = E_init
        self.embeddings_dtype = embeddings_dtype
        self.name = name

        super(EmbeddingInputlayer, self).__init__(E_init_args=E_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embeddings shape: %s" % self.embeddings_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("embeddings dtype: %s" % self.embeddings_dtype)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name):

            if self._temp_data['inputs'].dtype not in [tf.int32, tf.int64]:
                raise ValueError("The inputs of this layer should be of type: `tf.int32` or `tf.int64`")

            embeddings = self._get_tf_variable(
                name='embeddings',
                shape=self.embeddings_shape,
                dtype=self.embeddings_dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.E_init,
                **self.E_init_args
            )

            self._temp_data['outputs'] = tf.nn.embedding_lookup(
                params=embeddings,
                ids=self._temp_data['inputs'],
                partition_strategy='mod',
                name=None,
                validate_indices=True,
                max_norm=None
            )


class AverageEmbeddingInputlayer(Layer):
    """The :class:`AverageEmbeddingInputlayer` averages over embeddings of inputs.
    This is often used as the input layer for models like DAN[1] and FastText[2].

    Parameters
    ----------
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
        vocabulary_size=None,
        embedding_size=None,
        pad_value=0,
        embeddings_dtype=tf.float32,
        embeddings_initializer=tf.random_uniform_initializer(-0.1, 0.1),
        embeddings_kwargs=None,
        name='average_embedding',
    ):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings_shape = [self.vocabulary_size, self.embedding_size]

        self.pad_value = pad_value
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_dtype = embeddings_dtype
        self.name = name

        super(AverageEmbeddingInputlayer, self).__init__(embeddings_kwargs=embeddings_kwargs)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("embeddings shape: %s" % self.embeddings_shape)
        except AttributeError:
            pass

        try:
            additional_str.append("embeddings dtype: %s" % self.embeddings_dtype)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if self._temp_data['inputs'].get_shape().ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        with tf.variable_scope(self.name):

            embeddings = self._get_tf_variable(
                name='embeddings',
                shape=self.embeddings_shape,
                dtype=self.embeddings_dtype,
                initializer=self.embeddings_initializer,
                **self.embeddings_kwargs
            )

            word_embeddings = tf.nn.embedding_lookup(
                embeddings,
                self._temp_data['inputs'],
                name='word_embeddings',
            )
            # Zero out embeddings of pad value
            masks = tf.not_equal(self._temp_data['inputs'], self.pad_value, name='masks')

            word_embeddings *= tf.cast(tf.expand_dims(masks, axis=-1), dtype=self.embeddings_dtype)

            sum_word_embeddings = tf.reduce_sum(word_embeddings, axis=1)

            # Count number of non-padding words in each sentence
            sentence_lengths = tf.count_nonzero(
                masks,
                axis=1,
                keepdims=True,
                dtype=self.embeddings_dtype,
                name='sentence_lengths',
            )

            self._temp_data['outputs'] = tf.divide(
                sum_word_embeddings,
                sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
                name='sentence_embeddings'
            )
