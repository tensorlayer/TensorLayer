# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from .. import _logging as logging
from .. import files, iterate, utils, visualize

# __all__ = [
#     "Layer",
#     "DenseLayer",
# ]

# set_keep = locals()
set_keep = globals()
set_keep['_layers_name_list'] = []
set_keep['name_reuse'] = False

D_TYPE = tf.float32

try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except Exception:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES


def flatten_reshape(variable, name='flatten'):
    """Reshapes a high-dimension vector input.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row x mask_col x n_mask]

    Parameters
    ----------
    variable : TensorFlow variable or tensor
        The variable or tensor to be flatten.
    name : str
        A unique layer name.

    Returns
    -------
    Tensor
        Flatten Tensor

    Examples
    --------
    >>> W_conv2 = weight_variable([5, 5, 100, 32])   # 64 features for each 5x5 patch
    >>> b_conv2 = bias_variable([32])
    >>> W_fc1 = weight_variable([7 * 7 * 32, 256])

    >>> h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    >>> h_pool2 = max_pool_2x2(h_conv2)
    >>> h_pool2.get_shape()[:].as_list() = [batch_size, 7, 7, 32]
    ...         [batch_size, mask_row, mask_col, n_mask]
    >>> h_pool2_flat = tl.layers.flatten_reshape(h_pool2)
    ...         [batch_size, mask_row * mask_col * n_mask]
    >>> h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
    ...

    """
    dim = 1
    for d in variable.get_shape()[1:].as_list():
        dim *= d
    return tf.reshape(variable, shape=[-1, dim], name=name)


def clear_layers_name():
    """Clear all layer names in `set_keep['_layers_name_list']` if layer names are reused.

    Examples
    ---------
    Clean the current graph and try to re-define model.

    >>> for .... (different model settings):
    >>>    with tf.Graph().as_default() as graph:   # clear all variables of TF
    >>>       tl.layers.clear_layers_name()         # clear all layer name of TL
    >>>       sess = tf.InteractiveSession()
    >>>       # define and train a model here
    >>>       sess.close()

    Enable reusing layer names.

    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.DenseLayer(net, n_units=800, name='relu1')
    ...
    >>> tl.layers.clear_layers_name()
    >>> net2 = tl.layers.InputLayer(x, name='input_layer')
    >>> net2 = tl.layers.DenseLayer(net2, n_units=800, name='relu1')

    """
    set_keep['_layers_name_list'] = []


def set_name_reuse(enable=True):
    """Enable or disable reuse layer name.

    By default, each layer must has unique
    name. When you want two or more input placeholder (inference) share the same
    model parameters, you need to enable layer name reuse, then allow the
    parameters have same name scope.

    Parameters
    ----------
    enable : boolean
        Enable or disable name/layer reuse, None means False.

    Examples
    --------
    >>> def embed_seq(input_seqs, is_train, reuse):
    >>>    with tf.variable_scope("model", reuse=reuse):
    >>>         tl.layers.set_name_reuse(reuse)
    >>>         net = tl.layers.EmbeddingInputlayer(
    ...                     inputs = input_seqs,
    ...                     vocabulary_size = vocab_size,
    ...                     embedding_size = embedding_size,
    ...                     name = 'e_embedding')
    >>>         net = tl.layers.DynamicRNNLayer(net,
    ...                     cell_fn = tf.contrib.rnn.BasicLSTMCell,
    ...                     n_hidden = embedding_size,
    ...                     dropout = (0.7 if is_train else None),
    ...                     initializer = w_init,
    ...                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...                     return_last = True,
    ...                     name = 'e_dynamicrnn')
    >>>    return net
    >>>
    >>> net_train = embed_seq(t_caption, is_train=True, reuse=False)
    >>> net_test = embed_seq(t_caption, is_train=False, reuse=True)

    - see ``tutorial_ptb_lstm.py`` for example.

    """
    set_keep['name_reuse'] = enable


def initialize_rnn_state(state, feed_dict=None):
    """Returns the initialized RNN state.
    The inputs are `LSTMStateTuple` or `State` of `RNNCells`, and an optional `feed_dict`.

    Parameters
    ----------
    state : RNN state.
        The TensorFlow's RNN state.
    feed_dict : dictionary
        Initial RNN state; if None, returns zero state.

    Returns
    -------
    RNN state
        The TensorFlow's RNN state.

    """
    try:  # TF1.0
        LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple
    except Exception:
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

    if isinstance(state, LSTMStateTuple):
        c = state.c.eval(feed_dict=feed_dict)
        h = state.h.eval(feed_dict=feed_dict)
        return (c, h)
    else:
        new_state = state.eval(feed_dict=feed_dict)
        return new_state


def print_all_variables(train_only=False):
    """Print information of trainable or all variables,
    without ``tl.layers.initialize_global_variables(sess)``.

    Parameters
    ----------
    train_only : boolean
        Whether print trainable variables only.
            - If True, print the trainable variables.
            - If False, print all variables.

    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        logging.info("  [*] printing trainable variables")
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except Exception:  # TF0.12
            t_vars = tf.all_variables()
        logging.info("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        logging.info("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))


def get_variables_with_name(name=None, train_only=True, printable=False):
    """Get a list of TensorFlow variables by a given name scope.

    Parameters
    ----------
    name : str
        Get the variables that contain this name.
    train_only : boolean
        If Ture, only get the trainable variables.
    printable : boolean
        If True, print the information of all variables.

    Returns
    -------
    list of Tensor
        A list of TensorFlow variables

    Examples
    --------
    >>> dense_vars = tl.layers.get_variable_with_name('dense', True, True)

    """
    if name is None:
        raise Exception("please input a name")
    logging.info("  [*] geting variables with %s" % name)
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:  # TF1.0+
            t_vars = tf.global_variables()
        except Exception:  # TF0.12
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            logging.info("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


def get_layers_with_name(net, name="", printable=False):
    """Get a list of layers' output in a network by a given name scope.

    Parameters
    -----------
    net : :class:`Layer`
        The last layer of the network.
    name : str
        Get the layers' output that contain this name.
    printable : boolean
        If True, print information of all the layers' output

    Returns
    --------
    list of Tensor
        A list of layers' output (TensorFlow tensor)

    Examples
    ---------
    >>> layers = tl.layers.get_layers_with_name(net, "CNN", True)

    """
    logging.info("  [*] geting layers with %s" % name)

    layers = []
    i = 0
    for layer in net.all_layers:
        # logging.info(type(layer.name))
        if name in layer.name:
            layers.append(layer)
            if printable:
                logging.info("  got {:3}: {:15}   {}".format(i, layer.name, str(layer.get_shape())))
                i = i + 1
    return layers


def list_remove_repeat(x):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    x : list
        Input

    Returns
    -------
    list
        A list that after removing it's repeated items

    Examples
    -------
    >>> l = [2, 3, 4, 2, 3]
    >>> l = list_remove_repeat(l)
    ... [2, 3, 4]

    """
    y = []
    for i in x:
        if not i in y:
            y.append(i)
    return y


def merge_networks(layers=None):
    """Merge all parameters, layers and dropout probabilities to a :class:`Layer`.
    The output of return network is the first network in the list.

    Parameters
    ----------
    layers : list of :class:`Layer`
        Merge all parameters, layers and dropout probabilities to the first layer in the list.

    Returns
    --------
    :class:`Layer`
        The network after merging all parameters, layers and dropout probabilities to the first network in the list.

    Examples
    ---------
    >>> n1 = ...
    >>> n2 = ...
    >>> n1 = tl.layers.merge_networks([n1, n2])

    """
    if layers is None:
        raise Exception("layers should be a list of TensorLayer's Layers.")
    layer = layers[0]

    all_params = []
    all_layers = []
    all_drop = {}
    for l in layers:
        all_params.extend(l.all_params)
        all_layers.extend(l.all_layers)
        all_drop.update(l.all_drop)

    layer.all_params = list(all_params)
    layer.all_layers = list(all_layers)
    layer.all_drop = dict(all_drop)

    layer.all_layers = list_remove_repeat(layer.all_layers)
    layer.all_params = list_remove_repeat(layer.all_params)

    return layer


def initialize_global_variables(sess):
    """Initialize the global variables of TensorFlow.

    Run ``sess.run(tf.global_variables_initializer())`` for TF 0.12+ or
    ``sess.run(tf.initialize_all_variables())`` for TF 0.11.

    Parameters
    ----------
    sess : Session
        TensorFlow session.

    """
    assert sess is not None
    # try:    # TF12+
    sess.run(tf.global_variables_initializer())
    # except: # TF11
    #     sess.run(tf.initialize_all_variables())


class Layer(object):
    """
    The basic :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    inputs : :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : str or None
        A unique layer name.

    Methods
    ---------
    print_params(details=True, session=None)
        Print all parameters of this network.
    print_layers()
        Print all outputs of all layers of this network.
    count_params()
        Return the number of parameters of this network.
    """

    def __init__(self, inputs=None, name='layer'):
        self.inputs = inputs
        scope_name = tf.get_variable_scope().name
        if scope_name:
            name = scope_name + '/' + name
        if (name in set_keep['_layers_name_list']) and set_keep['name_reuse'] == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)\
            \nAdditional Informations: http://tensorlayer.readthedocs.io/en/latest/modules/layers.html?highlight=clear_layers_name#tensorlayer.layers.clear_layers_name"
                            % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)

    def print_params(self, details=True, session=None):
        """Print all info of parameters in the network"""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    # logging.info("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                    val = p.eval(session=session)
                    logging.info("  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                        i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()))
                except Exception as e:
                    logging.info(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network"""
        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info("  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name))

    def count_params(self):
        """Return the number of parameters in the network"""
        n_params = 0
        for _i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except Exception:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        return "  Last layer is: %s (%s) %s" % (self.__class__.__name__, self.name, self.outputs.get_shape().as_list())

    def __getitem__(self, key):
        set_name_reuse(True)
        net_new = Layer(self.inputs, name=self.name)
        set_name_reuse(set_keep['name_reuse'])  # set back
        net_new.outputs = self.outputs[key]

        net_new.all_layers = list(self.all_layers[:-1])
        net_new.all_layers.append(net_new.outputs)
        net_new.all_params = list(self.all_params)
        net_new.all_drop = dict(self.all_drop)
        return net_new

    def __setitem__(self, key, item):
        # self.outputs[key] = item
        raise NotImplementedError("%s: __setitem__" % self.name)

    def __delitem__(self, key):
        raise NotImplementedError("%s: __delitem__" % self.name)

    def __iter__(self):
        for x in self.all_layers:
            yield x

    def __len__(self):
        return len(self.all_layers)


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
        Layer.__init__(self, inputs=inputs, name=name)
        logging.info("InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}


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

    """

    def __init__(self, inputs=None, depth=None, on_value=None, off_value=None, axis=None, dtype=None, name='input'):
        Layer.__init__(self, inputs=inputs, name=name)
        logging.info("OneHotInputLayer  %s: %s" % (self.name, inputs.get_shape()))
        # assert depth != None, "depth is not given"
        if depth is None:
            logging.info("  [*] depth == None the number of output units is undefined")
        self.outputs = tf.one_hot(inputs, depth, on_value=on_value, off_value=off_value, axis=axis, dtype=dtype)
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}


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

    >>> train_inputs = tf.placeholder(tf.int32, shape=(batch_size))
    >>> train_labels = tf.placeholder(tf.int32, shape=(batch_size, 1))
    >>> emb_net = tl.layers.Word2vecEmbeddingInputlayer(
    ...         inputs = train_inputs,
    ...         train_labels = train_labels,
    ...         vocabulary_size = vocabulary_size,
    ...         embedding_size = embedding_size,
    ...         num_sampled = num_sampled,
    ...         name ='word2vec',
    ...    )
    >>> cost = emb_net.nce_cost
    >>> train_params = emb_net.all_params
    >>> train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    ...                                             cost, var_list=train_params)
    >>> normalized_embeddings = emb_net.normalized_embeddings

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
            name='word2vec',
    ):
        if nce_loss_args is None:
            nce_loss_args = {}
        if E_init_args is None:
            E_init_args = {}
        if nce_W_init_args is None:
            nce_W_init_args = {}
        if nce_b_init_args is None:
            nce_b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = inputs
        logging.info("Word2vecEmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.
        with tf.variable_scope(name):
            embeddings = tf.get_variable(name='embeddings', shape=(vocabulary_size, embedding_size), initializer=E_init, dtype=D_TYPE, **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)
            # Construct the variables for the NCE loss (i.e. negative sampling)
            nce_weights = tf.get_variable(name='nce_weights', shape=(vocabulary_size, embedding_size), initializer=nce_W_init, dtype=D_TYPE, **nce_W_init_args)
            nce_biases = tf.get_variable(name='nce_biases', shape=(vocabulary_size), initializer=nce_b_init, dtype=D_TYPE, **nce_b_init_args)

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels
        # each time we evaluate the loss.
        self.nce_cost = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                inputs=embed,
                labels=train_labels,
                num_sampled=num_sampled,
                num_classes=vocabulary_size,
                **nce_loss_args))

        self.outputs = embed
        self.normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)

        self.all_layers = [self.outputs]
        self.all_params = [embeddings, nce_weights, nce_biases]
        self.all_drop = {}


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
    >>> x = tf.placeholder(tf.int32, shape=(batch_size,))
    >>> emb_net = tl.layers.EmbeddingInputlayer(
    ...                inputs = x,
    ...                vocabulary_size = vocabulary_size,
    ...                embedding_size = embedding_size,
    ...                name ='embed')

    """

    def __init__(
            self,
            inputs=None,
            vocabulary_size=80000,
            embedding_size=200,
            E_init=tf.random_uniform_initializer(-0.1, 0.1),
            E_init_args=None,
            name='embedding',
    ):
        if E_init_args is None:
            E_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = inputs
        logging.info("EmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        with tf.variable_scope(name):
            embeddings = tf.get_variable(name='embeddings', shape=(vocabulary_size, embedding_size), initializer=E_init, dtype=D_TYPE, **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}


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

    """

    def __init__(
            self,
            inputs,
            vocabulary_size,
            embedding_size,
            pad_value=0,
            embeddings_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            embeddings_kwargs=None,
            name='average_embedding',
    ):
        super().__init__(name=name)

        # if embeddings_kwargs is None:
        #     embeddings_kwargs = {}

        if inputs.get_shape().ndims != 2:
            raise ValueError('inputs must be of size batch_size * batch_sentence_length')

        self.inputs = inputs

        logging.info("AverageEmbeddingInputlayer %s: (%d, %d)" % (name, vocabulary_size, embedding_size))
        with tf.variable_scope(name):
            self.embeddings = tf.get_variable(
                name='embeddings',
                shape=(vocabulary_size, embedding_size),
                initializer=embeddings_initializer,
                dtype=D_TYPE,
                **(embeddings_kwargs or {})
                # **embeddings_kwargs
            )  # **(embeddings_kwargs or {}),

            word_embeddings = tf.nn.embedding_lookup(
                self.embeddings,
                self.inputs,
                name='word_embeddings',
            )
            # Zero out embeddings of pad value
            masks = tf.not_equal(self.inputs, pad_value, name='masks')
            word_embeddings *= tf.cast(
                tf.expand_dims(masks, axis=-1),
                # tf.float32,
                dtype=D_TYPE,
            )
            sum_word_embeddings = tf.reduce_sum(word_embeddings, axis=1)

            # Count number of non-padding words in each sentence
            sentence_lengths = tf.count_nonzero(
                masks,
                axis=1,
                keep_dims=True,
                # dtype=tf.float32,
                dtype=D_TYPE,
                name='sentence_lengths',
            )

            sentence_embeddings = tf.divide(
                sum_word_embeddings,
                sentence_lengths + 1e-8,  # Add epsilon to avoid dividing by 0
                name='sentence_embeddings')

        self.outputs = sentence_embeddings
        self.all_layers = [self.outputs]
        self.all_params = [self.embeddings]
        self.all_drop = {}


class DenseLayer(Layer):
    """The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, 800, act=tf.nn.relu, name='relu')

    Without native TensorLayer APIs, you can do as follow.

    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`FlattenLayer`.

    """

    def __init__(
            self,
            layer,
            n_units=100,
            act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dense',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        logging.info("DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=D_TYPE, **b_init_args)
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class ReconLayer(DenseLayer):
    """A reconstruction layer for :class:`DenseLayer` to implement AutoEncoder.

    It is often used to pre-train the previous :class:`DenseLayer`

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    x_recon : placeholder or tensor
        The target for reconstruction.
    n_units : int
        The number of units of the layer. It should equal ``x_recon``.
    act : activation function
        The activation function of this layer.
        Normally, for sigmoid layer, the reconstruction activation is ``sigmoid``;
        for rectifying layer, the reconstruction activation is ``softplus``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.DenseLayer(net, n_units=196,
    ...                                 act=tf.nn.sigmoid, name='sigmoid1')
    >>> recon_layer1 = tl.layers.ReconLayer(net, x_recon=x, n_units=784,
    ...                                 act=tf.nn.sigmoid, name='recon_layer1')
    >>> recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    ...                         denoise_name=None, n_epoch=1200, batch_size=128,
    ...                         print_freq=10, save=True, save_name='w1pre_')

    Methods
    -------
    pretrain(sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre')
        Start to pre-train the parameters of the previous DenseLayer.

    Notes
    -----
    The input layer should be `DenseLayer` or a layer that has only one axes.
    You may need to modify this part to define your own cost function.
    By default, the cost is implemented as follow:
    - For sigmoid layer, the implementation can be `UFLDL <http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial>`__
    - For rectifying layer, the implementation can be `Glorot (2011). Deep Sparse Rectifier Neural Networks <http://doi.org/10.1.1.208.6449>`__

    """

    def __init__(
            self,
            layer,
            x_recon=None,
            n_units=784,
            act=tf.nn.softplus,
            name='recon',
    ):
        DenseLayer.__init__(self, layer=layer, n_units=n_units, act=act, name=name)
        logging.info("%s is a ReconLayer" % self.name)

        # y : reconstruction outputs; train_params : parameters to train
        # Note that: train_params = [W_encoder, b_encoder, W_decoder, b_encoder]
        y = self.outputs
        self.train_params = self.all_params[-4:]

        # =====================================================================
        #
        # You need to modify the below cost function and optimizer so as to
        # implement your own pre-train method.
        #
        # =====================================================================
        lambda_l2_w = 0.004
        learning_rate = 0.0001
        logging.info("     lambda_l2_w: %f" % lambda_l2_w)
        logging.info("     learning_rate: %f" % learning_rate)

        # Mean-square-error i.e. quadratic-cost
        mse = tf.reduce_sum(tf.squared_difference(y, x_recon), 1)
        mse = tf.reduce_mean(mse)  # in theano: mse = ((y - x) ** 2 ).sum(axis=1).mean()
        # mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y, x_recon)),  1))
        # mse = tf.reduce_mean(tf.squared_difference(y, x_recon)) # <haodong>: Error
        # mse = tf.sqrt(tf.reduce_mean(tf.square(y - x_recon)))   # <haodong>: Error
        # Cross-entropy
        # ce = cost.cross_entropy(y, x_recon)                                               # <haodong>: list , list , Error (only be used for softmax output)
        # ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x_recon))          # <haodong>: list , list , Error (only be used for softmax output)
        # ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, x_recon))   # <haodong>: list , index , Error (only be used for softmax output)
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[0]) \
                + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])           # faster than the code below
        # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))

        # DropNeuro
        # P_o = cost.lo_regularizer(0.03)(
        #     self.train_params[0])  # + cost.lo_regularizer(0.5)(self.train_params[2])    # <haodong>: if add lo on decoder, no neuron will be broken
        # P_i = cost.li_regularizer(0.03)(self.train_params[0])  # + cost.li_regularizer(0.001)(self.train_params[2])

        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(activation_out)  # <haodong>:  theano: T.mean( self.a[i] )         # some neuron are broken, white and black
        # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, 0) )         # <haodong>: some neuron are broken, white and black
        # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, 1) )   # <haodong>: some neuron are broken, white and black
        # KL Divergence
        beta = 4
        rho = 0.15
        p_hat = tf.reduce_mean(activation_out, 0)  # theano: p_hat = T.mean( self.a[i], axis=0 )
        try:  # TF1.0
            KLD = beta * tf.reduce_sum(rho * tf.log(tf.divide(rho, p_hat)) + (1 - rho) * tf.log((1 - rho) / (tf.subtract(float(1), p_hat))))
        except Exception:  # TF0.12
            KLD = beta * tf.reduce_sum(rho * tf.log(tf.div(rho, p_hat)) + (1 - rho) * tf.log((1 - rho) / (tf.sub(float(1), p_hat))))
            # KLD = beta * tf.reduce_sum( rho * tf.log(rho/ p_hat) + (1- rho) * tf.log((1- rho)/(1- p_hat)) )
            # theano: L1_a = l1_a[i] * T.sum( rho[i] * T.log(rho[i]/ p_hat) + (1- rho[i]) * T.log((1- rho[i])/(1- p_hat)) )
        # Total cost
        if act == tf.nn.softplus:
            logging.info('     use: mse, L2_w, L1_a')
            self.cost = mse + L1_a + L2_w
        elif act == tf.nn.sigmoid:
            # ----------------------------------------------------
            # Cross-entropy was used in Denoising AE
            # logging.info('     use: ce, L2_w, KLD')
            # self.cost = ce + L2_w + KLD
            # ----------------------------------------------------
            # Mean-squared-error was used in Vanilla AE
            logging.info('     use: mse, L2_w, KLD')
            self.cost = mse + L2_w + KLD
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_o) can remove neurons of AE
            # logging.info('     use: mse, L2_w, KLD, P_o')
            # self.cost = mse + L2_w + KLD + P_o
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_i) can remove neurons of previous layer
            #   If previous layer is InputLayer, it means remove useless features
            # logging.info('     use: mse, L2_w, KLD, P_i')
            # self.cost = mse + L2_w + KLD + P_i
        else:
            raise Exception("Don't support the given reconstruct activation function")

        self.train_op = tf.train.AdamOptimizer(
            learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(
                self.cost, var_list=self.train_params)
        # self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost, var_list=self.train_params)

    def pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_'):
        # ====================================================
        #
        # You need to modify the cost function in __init__() so as to
        # get your own pre-train method.
        #
        # ====================================================
        logging.info("     [*] %s start pretrain" % self.name)
        logging.info("     batch_size: %d" % batch_size)
        if denoise_name:
            logging.info("     denoising layer keep: %f" % self.all_drop[set_keep[denoise_name]])
            dp_denoise = self.all_drop[set_keep[denoise_name]]
        else:
            logging.info("     no denoising layer")

        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                dp_dict = utils.dict_to_one(self.all_drop)
                if denoise_name:
                    dp_dict[set_keep[denoise_name]] = dp_denoise
                feed_dict = {x: X_train_a}
                feed_dict.update(dp_dict)
                sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                logging.info("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_train_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch += 1
                logging.info("   train loss: %f" % (train_loss / n_batch))
                val_loss, n_batch = 0, 0
                for X_val_a, _ in iterate.minibatches(X_val, X_val, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one(self.all_drop)
                    feed_dict = {x: X_val_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1
                logging.info("   val loss: %f" % (val_loss / n_batch))
                if save:
                    try:
                        visualize.draw_weights(
                            self.train_params[0].eval(), second=10, saveable=True, shape=[28, 28], name=save_name + str(epoch + 1), fig_idx=2012)
                        files.save_npz([self.all_params[0]], name=save_name + str(epoch + 1) + '.npz')
                    except Exception:
                        raise Exception(
                            "You should change the visualize.W() in ReconLayer.pretrain(), if you want to save the feature images for different dataset")


class DropoutLayer(Layer):
    """
    The :class:`DropoutLayer` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    is_fix : boolean
        Fixing probability or nor. Default is False.
        If True, the keeping probability is fixed and cannot be changed via `feed_dict`.
    is_train : boolean
        Trainable or not. If False, skip this layer. Default is True.
    seed : int or None
        The seed for random dropout.
    name : str
        A unique layer name.

    Examples
    --------
    Method 1: Using ``all_drop`` see `tutorial_mlp_dropout1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py>`__

    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.DropoutLayer(net, keep=0.8, name='drop1')
    >>> net = tl.layers.DenseLayer(net, n_units=800, act=tf.nn.relu, name='relu1')
    >>> ...
    >>> # For training, enable dropout as follow.
    >>> feed_dict = {x: X_train_a, y_: y_train_a}
    >>> feed_dict.update( net.all_drop )     # enable noise layers
    >>> sess.run(train_op, feed_dict=feed_dict)
    >>> ...
    >>> # For testing, disable dropout as follow.
    >>> dp_dict = tl.utils.dict_to_one( net.all_drop ) # disable noise layers
    >>> feed_dict = {x: X_val_a, y_: y_val_a}
    >>> feed_dict.update(dp_dict)
    >>> err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    >>> ...

    Method 2: Without using ``all_drop`` see `tutorial_mlp_dropout2.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py>`__

    >>> def mlp(x, is_train=True, reuse=False):
    >>>     with tf.variable_scope("MLP", reuse=reuse):
    >>>     tl.layers.set_name_reuse(reuse)
    >>>     net = tl.layers.InputLayer(x, name='input')
    >>>     net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True,
    >>>                         is_train=is_train, name='drop1')
    >>>     ...
    >>>     return net
    >>> # define inferences
    >>> net_train = mlp(x, is_train=True, reuse=False)
    >>> net_test = mlp(x, is_train=False, reuse=True)

    """

    def __init__(
            self,
            layer,
            keep=0.5,
            is_fix=False,
            is_train=True,
            seed=None,
            name='dropout_layer',
    ):
        Layer.__init__(self, name=name)
        if is_train is False:
            logging.info("skip DropoutLayer")
            self.outputs = layer.outputs
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
        else:
            self.inputs = layer.outputs
            logging.info("DropoutLayer %s: keep:%f is_fix:%s" % (self.name, keep, is_fix))

            # The name of placeholder for keep_prob is the same with the name
            # of the Layer.
            if is_fix:
                self.outputs = tf.nn.dropout(self.inputs, keep, seed=seed, name=name)
            else:
                set_keep[name] = tf.placeholder(tf.float32)
                self.outputs = tf.nn.dropout(self.inputs, set_keep[name], seed=seed, name=name)  # 1.2

            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
            if is_fix is False:
                self.all_drop.update({set_keep[name]: keep})
            self.all_layers.extend([self.outputs])

        # logging.info(set_keep[name])
        #   Tensor("Placeholder_2:0", dtype=float32)
        # logging.info(denoising1)
        #   Tensor("Placeholder_2:0", dtype=float32)
        # logging.info(self.all_drop[denoising1])
        #   0.8
        #
        # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/tf/index.html
        # The optional feed_dict argument allows the caller to override the
        # value of tensors in the graph. Each key in feed_dict can be one of
        # the following types:
        # If the key is a Tensor, the value may be a Python scalar, string,
        # list, or numpy ndarray that can be converted to the same dtype as that
        # tensor. Additionally, if the key is a placeholder, the shape of the
        # value will be checked for compatibility with the placeholder.
        # If the key is a SparseTensor, the value should be a SparseTensorValue.


class GaussianNoiseLayer(Layer):
    """
    The :class:`GaussianNoiseLayer` class is noise layer that adding noise with
    gaussian distribution to the activation.

    Parameters
    ------------
    layer : :class:`Layer`
        Previous layer.
    mean : float
        The mean. Default is 0.
    stddev : float
        The standard deviation. Default is 1.
    is_train : boolean
        Is trainable layer. If False, skip this layer. default is True.
    seed : int or None
        The seed for random noise.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
            mean=0.0,
            stddev=1.0,
            is_train=True,
            seed=None,
            name='gaussian_noise_layer',
    ):
        Layer.__init__(self, name=name)
        if is_train is False:
            logging.info("skip GaussianNoiseLayer")
            self.outputs = layer.outputs
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
        else:
            self.inputs = layer.outputs
            logging.info("GaussianNoiseLayer %s: mean:%f stddev:%f" % (self.name, mean, stddev))
            with tf.variable_scope(name):
                # noise = np.random.normal(0.0 , sigma , tf.to_int64(self.inputs).get_shape())
                noise = tf.random_normal(shape=self.inputs.get_shape(), mean=mean, stddev=stddev, seed=seed)
                self.outputs = self.inputs + noise
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)


class DropconnectDenseLayer(Layer):
    """
    The :class:`DropconnectDenseLayer` class is :class:`DenseLayer` with DropConnect
    behaviour which randomly removes connections between this layer and the previous
    layer according to a keeping probability.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : weights initializer
        The initializer for the weight matrix.
    b_init : biases initializer
        The initializer for the bias vector.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.InputLayer(x, name='input_layer')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.8,
    ...         n_units=800, act=tf.nn.relu, name='relu1')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.5,
    ...         n_units=800, act=tf.nn.relu, name='relu2')
    >>> net = tl.layers.DropconnectDenseLayer(net, keep=0.5,
    ...         n_units=10, name='output')

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`__

    """

    def __init__(
            self,
            layer,
            keep=0.5,
            n_units=100,
            act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dropconnect_layer',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2")
        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        logging.info("DropconnectDenseLayer %s: %d %s" % (self.name, self.n_units, act.__name__))

        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=D_TYPE, **W_init_args)
            b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=D_TYPE, **b_init_args)
            self.outputs = act(tf.matmul(self.inputs, W) + b)

        set_keep[name] = tf.placeholder(tf.float32)
        W_dropcon = tf.nn.dropout(W, set_keep[name])
        self.outputs = act(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_drop.update({set_keep[name]: keep})
        self.all_layers.extend([self.outputs])
        self.all_params.extend([W, b])
