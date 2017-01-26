#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import time
from . import visualize
from . import utils
from . import files
from . import cost
from . import iterate
import numpy as np
from six.moves import xrange
import random
import warnings

# __all__ = [
#     "Layer",
#     "DenseLayer",
# ]


# set_keep = locals()
set_keep = globals()
set_keep['_layers_name_list'] =[]
set_keep['name_reuse'] = False

try:  # For TF12 and later
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
except:  # For TF11 and before
    TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.VARIABLES

## Variable Operation
def flatten_reshape(variable, name=''):
    """Reshapes high-dimension input to a vector.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    variable : a tensorflow variable
    name : a string or None
        An optional name to attach to this layer.

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
    """Clear all layer names in set_keep['_layers_name_list'],
    enable layer name reuse.

    Examples
    ---------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(network, n_units=800, name='relu1')
    ...
    >>> tl.layers.clear_layers_name()
    >>> network2 = tl.layers.InputLayer(x, name='input_layer')
    >>> network2 = tl.layers.DenseLayer(network2, n_units=800, name='relu1')
    ...
    """
    set_keep['_layers_name_list'] =[]

def set_name_reuse(enable=True):
    """Enable or disable reuse layer name. By default, each layer must has unique
    name. When you want two or more input placeholder (inference) share the same
    model parameters, you need to enable layer name reuse, then allow the
    parameters have same name scope.

    Parameters
    ------------
    enable : boolean, enable name reuse.

    Examples
    ------------
    >>> def embed_seq(input_seqs, is_train, reuse):
    >>>    with tf.variable_scope("model", reuse=reuse):
    >>>         tl.layers.set_name_reuse(reuse)
    >>>         network = tl.layers.EmbeddingInputlayer(
    ...                     inputs = input_seqs,
    ...                     vocabulary_size = vocab_size,
    ...                     embedding_size = embedding_size,
    ...                     name = 'e_embedding')
    >>>        network = tl.layers.DynamicRNNLayer(network,
    ...                     cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
    ...                     n_hidden = embedding_size,
    ...                     dropout = (0.7 if is_train else None),
    ...                     initializer = w_init,
    ...                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...                     return_last = True,
    ...                     name = 'e_dynamicrnn',)
    >>>    return network
    >>>
    >>> net_train = embed_seq(t_caption, is_train=True, reuse=False)
    >>> net_test = embed_seq(t_caption, is_train=False, reuse=True)

    - see ``tutorial_ptb_lstm.py`` for example.
    """
    set_keep['name_reuse'] = enable

def initialize_rnn_state(state):
    """Return the initialized RNN state.
    The input is LSTMStateTuple or State of RNNCells.

    Parameters
    -----------
    state : a RNN state.
    """
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        c = state.c.eval()
        h = state.h.eval()
        return (c, h)
    else:
        new_state = state.eval()
        return new_state


def print_all_variables(train_only=False):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    tvar = tf.trainable_variables() if train_only else tf.all_variables()
    for idx, v in enumerate(tvar):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))


def get_variables_with_name(name, train_only=True, printable=False):
    """Get variable list by a given name scope.

    Examples
    ---------
    >>> dense_vars = get_variable_with_name('dense', True, True)
    """
    print("  Get variables with %s" % name)
    t_vars = tf.trainable_variables() if train_only else tf.all_variables()
    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars

def list_remove_repeat(l=None):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    l : a list

    Examples
    ---------
    >>> l = [2, 3, 4, 2, 3]
    >>> l = list_remove_repeat(l)
    ... [2, 3, 4]
    """
    l2 = []
    [l2.append(i) for i in l if not i in l2]
    return l2


def initialize_global_variables(sess=None):
    """Excute ``sess.run(tf.global_variables_initializer())`` for TF12+ or
    sess.run(tf.initialize_all_variables()) for TF11.

    Parameters
    ----------
    sess : a Session
    """
    assert sess is not None
    try:    # TF12
        sess.run(tf.global_variables_initializer())
    except: # TF11
        sess.run(tf.initialize_all_variables())


## Basic layer
class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    inputs : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        inputs = None,
        name ='layer'
    ):
        self.inputs = inputs
        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name'.\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)


    def print_params(self, details=True):
        ''' Print all info of parameters in the network'''
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                except Exception as e:
                    print(str(e))
                    raise Exception("Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(i, str(p.get_shape()), p.name))
        print("  num of params: %d" % self.count_params())

    def print_layers(self):
        ''' Print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            print("  layer %d: %s" % (i, str(p)))

    def count_params(self):
        ''' Return the number of parameters in the network '''
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        print("\nIt is a Layer class")
        self.print_params(False)
        self.print_layers()
        return "  Last layer is: %s" % self.__class__.__name__

## Input layer
class InputLayer(Layer):
    """
    The :class:`InputLayer` class is the starting layer of a neural network.

    Parameters
    ----------
    inputs : a TensorFlow placeholder
        The input tensor data.
    name : a string or None
        An optional name to attach to this layer.
    n_features : a int
        The number of features. If not specify, it will assume the input is
        with the shape of [batch_size, n_features], then select the second
        element as the n_features. It is used to specify the matrix size of
        next layer. If apply Convolutional layer after InputLayer,
        n_features is not important.
    """
    def __init__(
        self,
        inputs = None,
        n_features = None,
        name ='input_layer'
    ):
        Layer.__init__(self, inputs=inputs, name=name)
        print("  tensorlayer:Instantiate InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}

## Word Embedding Input layer
class Word2vecEmbeddingInputlayer(Layer):
    """
    The :class:`Word2vecEmbeddingInputlayer` class is a fully connected layer,
    for Word Embedding. Words are input as integer index.
    The output is the embedded word vector.

    Parameters
    ----------
    inputs : placeholder
        For word inputs. integer index format.
    train_labels : placeholder
        For word labels. integer index format.
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    num_sampled : int
        The Number of negative examples for NCE loss.
    nce_loss_args : a dictionary
        The arguments for tf.nn.nce_loss()
    E_init : embedding initializer
        The initializer for initializing the embedding matrix.
    E_init_args : a dictionary
        The arguments for embedding initializer
    nce_W_init : NCE decoder biases initializer
        The initializer for initializing the nce decoder weight matrix.
    nce_W_init_args : a dictionary
        The arguments for initializing the nce decoder weight matrix.
    nce_b_init : NCE decoder biases initializer
        The initializer for tf.get_variable() of the nce decoder bias vector.
    nce_b_init_args : a dictionary
        The arguments for tf.get_variable() of the nce decoder bias vector.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    nce_cost : a tensor
        The NCE loss.
    outputs : a tensor
        The outputs of embedding layer.
    normalized_embeddings : tensor
        Normalized embedding matrix

    Examples
    --------
    - Without TensorLayer : see tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    >>> train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    >>> train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
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

    - With TensorLayer : see tutorial_word2vec_basic.py
    >>> train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    >>> train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    >>> emb_net = tl.layers.Word2vecEmbeddingInputlayer(
    ...         inputs = train_inputs,
    ...         train_labels = train_labels,
    ...         vocabulary_size = vocabulary_size,
    ...         embedding_size = embedding_size,
    ...         num_sampled = num_sampled,
    ...        name ='word2vec_layer',
    ...    )
    >>> cost = emb_net.nce_cost
    >>> train_params = emb_net.all_params
    >>> train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    ...                                             cost, var_list=train_params)
    >>> normalized_embeddings = emb_net.normalized_embeddings

    References
    ----------
    - `tensorflow/examples/tutorials/word2vec/word2vec_basic.py <https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>`_
    """
    def __init__(
        self,
        inputs = None,
        train_labels = None,
        vocabulary_size = 80000,
        embedding_size = 200,
        num_sampled = 64,
        nce_loss_args = {},
        E_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
        E_init_args = {},
        nce_W_init = tf.truncated_normal_initializer(stddev=0.03),
        nce_W_init_args = {},
        nce_b_init = tf.constant_initializer(value=0.0),
        nce_b_init_args = {},
        name ='word2vec_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs
        print("  tensorlayer:Instantiate Word2vecEmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))
        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.
        with tf.variable_scope(name) as vs:
            embeddings = tf.get_variable(name='embeddings',
                                    shape=(vocabulary_size, embedding_size),
                                    initializer=E_init,
                                    **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)
            # Construct the variables for the NCE loss (i.e. negative sampling)
            nce_weights = tf.get_variable(name='nce_weights',
                                    shape=(vocabulary_size, embedding_size),
                                    initializer=nce_W_init,
                                    **nce_W_init_args)
            nce_biases = tf.get_variable(name='nce_biases',
                                    shape=(vocabulary_size),
                                    initializer=nce_b_init,
                                    **nce_b_init_args)

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels
        # each time we evaluate the loss.
        self.nce_cost = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                           inputs=embed, labels=train_labels,
                           num_sampled=num_sampled, num_classes=vocabulary_size,
                           **nce_loss_args))

        self.outputs = embed
        self.normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)

        self.all_layers = [self.outputs]
        self.all_params = [embeddings, nce_weights, nce_biases]
        self.all_drop = {}

class EmbeddingInputlayer(Layer):
    """
    The :class:`EmbeddingInputlayer` class is a fully connected layer,
    for Word Embedding. Words are input as integer index.
    The output is the embedded word vector.

    If you have a pre-train matrix, you can assign the matrix into it.
    To train a word embedding matrix, you can used class:`Word2vecEmbeddingInputlayer`.

    Note that, do not update this embedding matrix.

    Parameters
    ----------
    inputs : placeholder
        For word inputs. integer index format.
        a 2D tensor : [batch_size, num_steps(num_words)]
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    E_init : embedding initializer
        The initializer for initializing the embedding matrix.
    E_init_args : a dictionary
        The arguments for embedding initializer
    name : a string or None
        An optional name to attach to this layer.

    Variables
    ------------
    outputs : a tensor
        The outputs of embedding layer.
        the outputs 3D tensor : [batch_size, num_steps(num_words), embedding_size]

    Examples
    --------
    >>> vocabulary_size = 50000
    >>> embedding_size = 200
    >>> model_file_name = "model_word2vec_50k_200"
    >>> batch_size = None
    ...
    >>> all_var = tl.files.load_npy_to_any(name=model_file_name+'.npy')
    >>> data = all_var['data']; count = all_var['count']
    >>> dictionary = all_var['dictionary']
    >>> reverse_dictionary = all_var['reverse_dictionary']
    >>> tl.files.save_vocab(count, name='vocab_'+model_file_name+'.txt')
    >>> del all_var, data, count
    ...
    >>> load_params = tl.files.load_npz(name=model_file_name+'.npz')
    >>> x = tf.placeholder(tf.int32, shape=[batch_size])
    >>> y_ = tf.placeholder(tf.int32, shape=[batch_size, 1])
    >>> emb_net = tl.layers.EmbeddingInputlayer(
    ...                inputs = x,
    ...                vocabulary_size = vocabulary_size,
    ...                embedding_size = embedding_size,
    ...                name ='embedding_layer')
    >>> tl.layers.initialize_global_variables(sess)
    >>> tl.files.assign_params(sess, [load_params[0]], emb_net)
    >>> word = b'hello'
    >>> word_id = dictionary[word]
    >>> print('word_id:', word_id)
    ... 6428
    ...
    >>> words = [b'i', b'am', b'hao', b'dong']
    >>> word_ids = tl.files.words_to_word_ids(words, dictionary)
    >>> context = tl.files.word_ids_to_words(word_ids, reverse_dictionary)
    >>> print('word_ids:', word_ids)
    ... [72, 1226, 46744, 20048]
    >>> print('context:', context)
    ... [b'i', b'am', b'hao', b'dong']
    ...
    >>> vector = sess.run(emb_net.outputs, feed_dict={x : [word_id]})
    >>> print('vector:', vector.shape)
    ... (1, 200)
    >>> vectors = sess.run(emb_net.outputs, feed_dict={x : word_ids})
    >>> print('vectors:', vectors.shape)
    ... (4, 200)

    """
    def __init__(
        self,
        inputs = None,
        vocabulary_size = 80000,
        embedding_size = 200,
        E_init = tf.random_uniform_initializer(-0.1, 0.1),
        E_init_args = {},
        name ='embedding_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs
        print("  tensorlayer:Instantiate EmbeddingInputlayer %s: (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        with tf.variable_scope(name) as vs:
            embeddings = tf.get_variable(name='embeddings',
                                    shape=(vocabulary_size, embedding_size),
                                    initializer=E_init,
                                    **E_init_args)
            embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}


## Dense layer
class DenseLayer(Layer):
    """
    The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_units : int
        The number of units of the layer.
    act : activation function
        The function that is applied to the layer activations.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable.
    b_init_args : dictionary
        The arguments for the biases tf.get_variable.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(
    ...                 network,
    ...                 n_units=800,
    ...                 act = tf.nn.relu,
    ...                 W_init=tf.truncated_normal_initializer(stddev=0.1),
    ...                 name ='relu_layer'
    ...                 )

    >>> Without TensorLayer, you can do as follow.
    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the input to this layer has more than two axes, it need to flatten the
    input by using :class:`FlattenLayer` in this case.
    """
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.1),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  tensorlayer:Instantiate DenseLayer  %s: %d, %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )

class ReconLayer(DenseLayer):
    """
    The :class:`ReconLayer` class is a reconstruction layer `DenseLayer` which
    use to pre-train a `DenseLayer`.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    x_recon : tensorflow variable
        The variables used for reconstruction.
    name : a string or None
        An optional name to attach to this layer.
    n_units : int
        The number of units of the layer, should be equal to x_recon
    act : activation function
        The activation function that is applied to the reconstruction layer.
        Normally, for sigmoid layer, the reconstruction activation is sigmoid;
        for rectifying layer, the reconstruction activation is softplus.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(network, n_units=196,
    ...                                 act=tf.nn.sigmoid, name='sigmoid1')
    >>> recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784,
    ...                                 act=tf.nn.sigmoid, name='recon_layer1')
    >>> recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val,
    ...                         denoise_name=None, n_epoch=1200, batch_size=128,
    ...                         print_freq=10, save=True, save_name='w1pre_')

    Methods
    -------
    pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
        Start to pre-train the parameters of previous DenseLayer.

    Notes
    -----
    The input layer should be `DenseLayer` or a layer has only one axes.
    You may need to modify this part to define your own cost function.
    By default, the cost is implemented as follow:
    - For sigmoid layer, the implementation can be `UFLDL <http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial>`_
    - For rectifying layer, the implementation can be `Glorot (2011). Deep Sparse Rectifier Neural Networks <http://doi.org/10.1.1.208.6449>`_
    """
    def __init__(
        self,
        layer = None,
        x_recon = None,
        name = 'recon_layer',
        n_units = 784,
        act = tf.nn.softplus,
    ):
        DenseLayer.__init__(self, layer=layer, n_units=n_units, act=act, name=name)
        print("     tensorlayer: %s is a ReconLayer" % self.name)

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
        print("     lambda_l2_w: %f" % lambda_l2_w)
        print("     learning_rate: %f" % learning_rate)

        # Mean-squre-error i.e. quadratic-cost
        mse = tf.reduce_sum(tf.squared_difference(y, x_recon), reduction_indices = 1)
        mse = tf.reduce_mean(mse)            # in theano: mse = ((y - x) ** 2 ).sum(axis=1).mean()
            # mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y, x_recon)), reduction_indices = 1))
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
        P_o = cost.lo_regularizer(0.03)(self.train_params[0])   # + cost.lo_regularizer(0.5)(self.train_params[2])    # <haodong>: if add lo on decoder, no neuron will be broken
        P_i = cost.li_regularizer(0.03)(self.train_params[0])  # + cost.li_regularizer(0.001)(self.train_params[2])

        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(activation_out)   # <haodong>:  theano: T.mean( self.a[i] )         # some neuron are broken, white and black
            # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=0) )         # <haodong>: some neuron are broken, white and black
            # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=1) )   # <haodong>: some neuron are broken, white and black
        # KL Divergence
        beta = 4
        rho = 0.15
        p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)   # theano: p_hat = T.mean( self.a[i], axis=0 )
        KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat)) + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )
            # KLD = beta * tf.reduce_sum( rho * tf.log(rho/ p_hat) + (1- rho) * tf.log((1- rho)/(1- p_hat)) )
            # theano: L1_a = l1_a[i] * T.sum( rho[i] * T.log(rho[i]/ p_hat) + (1- rho[i]) * T.log((1- rho[i])/(1- p_hat)) )
        # Total cost
        if act == tf.nn.softplus:
            print('     use: mse, L2_w, L1_a')
            self.cost = mse + L1_a + L2_w
        elif act == tf.nn.sigmoid:
            # ----------------------------------------------------
            # Cross-entropy was used in Denoising AE
            # print('     use: ce, L2_w, KLD')
            # self.cost = ce + L2_w + KLD
            # ----------------------------------------------------
            # Mean-squared-error was used in Vanilla AE
            print('     use: mse, L2_w, KLD')
            self.cost = mse + L2_w + KLD
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_o) can remove neurons of AE
            # print('     use: mse, L2_w, KLD, P_o')
            # self.cost = mse + L2_w + KLD + P_o
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_i) can remove neurons of previous layer
            #   If previous layer is InputLayer, it means remove useless features
            # print('     use: mse, L2_w, KLD, P_i')
            # self.cost = mse + L2_w + KLD + P_i
        else:
            raise Exception("Don't support the given reconstruct activation function")

        self.train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08, use_locking=False).minimize(self.cost, var_list=self.train_params)
                # self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost, var_list=self.train_params)

    def pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10,
                  save=True, save_name='w1pre_'):
        # ====================================================
        #
        # You need to modify the cost function in __init__() so as to
        # get your own pre-train method.
        #
        # ====================================================
        print("     tensorlayer:  %s start pretrain" % self.name)
        print("     batch_size: %d" % batch_size)
        if denoise_name:
            print("     denoising layer keep: %f" % self.all_drop[set_keep[denoise_name]])
            dp_denoise = self.all_drop[set_keep[denoise_name]]
        else:
            print("     no denoising layer")

        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                dp_dict = utils.dict_to_one( self.all_drop )
                if denoise_name:
                    dp_dict[set_keep[denoise_name]] = dp_denoise
                feed_dict = {x: X_train_a}
                feed_dict.update(dp_dict)
                sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one( self.all_drop )
                    feed_dict = {x: X_train_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch += 1
                print("   train loss: %f" % (train_loss/ n_batch))
                val_loss, n_batch = 0, 0
                for X_val_a, _ in iterate.minibatches(X_val, X_val, batch_size, shuffle=True):
                    dp_dict = utils.dict_to_one( self.all_drop )
                    feed_dict = {x: X_val_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                if save:
                    try:
                        visualize.W(self.train_params[0].eval(), second=10, saveable=True, shape=[28,28], name=save_name+str(epoch+1), fig_idx=2012)
                        files.save_npz([self.all_params[0]] , name=save_name+str(epoch+1)+'.npz')
                    except:
                        raise Exception("You should change the visualize.W() in ReconLayer.pretrain(), if you want to save the feature images for different dataset")


## Noise layer
class DropoutLayer(Layer):
    """
    The :class:`DropoutLayer` class is a noise layer which randomly set some
    values to zero by a given keeping probability.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    keep : float
        The keeping probability, the lower more values will be set to zero.
    is_fix : boolean
        Default False, if True, the keeping probability is fixed and cannot be changed via feed_dict.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - Define network
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    >>> network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
    >>> ...

    - For training, enable dropout as follow.
    >>> feed_dict = {x: X_train_a, y_: y_train_a}
    >>> feed_dict.update( network.all_drop )     # enable noise layers
    >>> sess.run(train_op, feed_dict=feed_dict)
    >>> ...

    - For testing, disable dropout as follow.
    >>> dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable noise layers
    >>> feed_dict = {x: X_val_a, y_: y_val_a}
    >>> feed_dict.update(dp_dict)
    >>> err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    >>> ...

    Notes
    -------
    - A frequent question regarding :class:`DropoutLayer` is that why it donot have `is_train` like :class:`BatchNormLayer`.
    In many simple cases, user may find it is better to use one inference instead of two inferences for training and testing seperately, :class:`DropoutLayer`
    allows you to control the dropout rate via `feed_dict`. However, you can fix the keeping probability by setting `is_fix` to True.
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        is_fix = False,
        name = 'dropout_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate DropoutLayer %s: keep: %f is_fix: %s" % (self.name, keep, is_fix))

        # The name of placeholder for keep_prob is the same with the name
        # of the Layer.
        if is_fix:
            self.outputs = tf.nn.dropout(self.inputs, keep, name=name)
        else:
            set_keep[name] = tf.placeholder(tf.float32)
            self.outputs = tf.nn.dropout(self.inputs, set_keep[name], name=name) # 1.2

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        if is_fix is False:
            self.all_drop.update( {set_keep[name]: keep} )
        self.all_layers.extend( [self.outputs] )

        # print(set_keep[name])
        #   Tensor("Placeholder_2:0", dtype=float32)
        # print(denoising1)
        #   Tensor("Placeholder_2:0", dtype=float32)
        # print(self.all_drop[denoising1])
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
    normal distribution to the activation.

    Parameters
    ------------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    sigma : float
        Scale value of gaussian noise.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        sigma = 0.1,
        name = 'gaussian_noise_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate GaussianNoiseLayer %s: keep: %f" % (self.name, keep))
        with tf.variable_scope(name) as vs:
            noise = np.random.normal(0.0 , sigma , tf.to_int64(input_layer).get_shape())
            self.inputs = self.inputs + noise
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)


class DropconnectDenseLayer(Layer):
    """
    The :class:`DropconnectDenseLayer` class is ``DenseLayer`` with DropConnect
    behaviour which randomly remove connection between this layer to previous
    layer by a given keeping probability.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    keep : float
        The keeping probability, the lower more values will be set to zero.
    n_units : int
        The number of units of the layer.
    act : activation function
        The function that is applied to the layer activations.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer
        The initializer for initializing the bias vector.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.8,
    ...         n_units=800, act = tf.nn.relu, name='dropconnect_relu1')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
    ...         n_units=800, act = tf.nn.relu, name='dropconnect_relu2')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
    ...         n_units=10, act = tl.activation.identity, name='output_layer')

    References
    ----------
    - `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`_
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        n_units = 100,
        act = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.1),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='dropconnect_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2")
        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  tensorlayer:Instantiate DropconnectDenseLayer %s: %d, %s" % (self.name, self.n_units, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
            b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
            self.outputs = act(tf.matmul(self.inputs, W) + b)#, name=name)    # 1.2

        set_keep[name] = tf.placeholder(tf.float32)
        W_dropcon = tf.nn.dropout(W,  set_keep[name])
        self.outputs = act(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_drop.update( {set_keep[name]: keep} )
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )


## Convolutional layer (Pro)

class Conv1dLayer(Layer):
    """
    The :class:`Conv1dLayer` class is a 1D CNN layer, see `tf.nn.conv1d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv1d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer, [batch, in_width, in_channels].
    act : activation function, None for identity.
    shape : list of shape
        shape of the filters, [filter_length, in_channels, out_channels].
    strides : an int.
        The number of entries by which the filter is moved right at each step.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    use_cudnn_on_gpu : An optional bool. Defaults to True.
    data_format : An optional string from "NHWC", "NCHW". Defaults to "NHWC", the data is stored in the order of [batch, in_width, in_channels]. The "NCHW" format stores data as [batch, in_channels, in_width].
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        act = tf.identity,
        shape = [5, 5, 1],
        strides= 1,
        padding='SAME',
        use_cudnn_on_gpu=None,
        data_format=None,
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate Conv1dLayer %s: %s, %s, %s, %s" %
                            (self.name, str(shape), str(strides), padding, act.__name__))
        if act is None:
            act = tf.identity
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_conv1d', shape=shape, initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b_conv1d', shape=(shape[-1]), initializer=b_init, **b_init_args )
                self.outputs = act( tf.nn.conv1d(self.inputs, W, stride=strides, padding=padding,
                            use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b ) #1.2
            else:
                self.outputs = act( tf.nn.conv1d(self.inputs, W, strides=strides, padding=padding,
                            use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )

class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    name : a string or None
        An optional name to attach to this layer.

    Notes
    ------
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                   act = tf.nn.relu,
    ...                   shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    ...                   strides=[1, 1, 1, 1],
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   W_init_args={},
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   b_init_args = {},
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> network = tl.layers.PoolLayer(network,
    ...                   ksize=[1, 2, 2, 1],
    ...                   strides=[1, 2, 2, 1],
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    >>> Without TensorLayer, you can implement 2d convolution as follow.
    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )
    """
    def __init__(
        self,
        layer = None,
        act = tf.identity,
        shape = [5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate Conv2dLayer %s: %s, %s, %s, %s" %
                            (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, **b_init_args )
                self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding) + b ) #1.2
            else:
                self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )

class DeConv2dLayer(Layer):
    """
    The :class:`DeConv2dLayer` class is deconvolutional 2D layer, see `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d_transpose>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [height, width, output_channels, in_channels], filter's in_channels dimension must match that of value.
    output_shape : list of output shape
        representing the output shape of the deconvolution op.
    strides : a list of ints.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights initializer.
    b_init_args : dictionary
        The arguments for the biases initializer.
    name : a string or None
        An optional name to attach to this layer.

    Notes
    -----
    - shape = [h, w, the number of output channels of this layer, the number of output channel of previous layer]
    - output_shape = [batch_size, any, any, the number of output channels of this layer]
    - the number of output channel of a layer is its last dimension.

    Examples
    ---------
    - A part of the generator in DCGAN example
    >>> batch_size = 64
    >>> inputs = tf.placeholder(tf.float32, [batch_size, 100], name='z_noise')
    >>> net_in = tl.layers.InputLayer(inputs, name='g/in')
    >>> net_h0 = tl.layers.DenseLayer(net_in, n_units = 8192,
    ...                            W_init = tf.random_normal_initializer(stddev=0.02),
    ...                            act = tf.identity, name='g/h0/lin')
    >>> print(net_h0.outputs._shape)
    ... (64, 8192)
    >>> net_h0 = tl.layers.ReshapeLayer(net_h0, shape = [-1, 4, 4, 512], name='g/h0/reshape')
    >>> net_h0 = tl.layers.BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train, name='g/h0/batch_norm')
    >>> print(net_h0.outputs._shape)
    ... (64, 4, 4, 512)
    >>> net_h1 = tl.layers.DeConv2dLayer(net_h0,
    ...                            shape = [5, 5, 256, 512],
    ...                            output_shape = [batch_size, 8, 8, 256],
    ...                            strides=[1, 2, 2, 1],
    ...                            act=tf.identity, name='g/h1/decon2d')
    >>> net_h1 = tl.layers.BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train, name='g/h1/batch_norm')
    >>> print(net_h1.outputs._shape)
    ... (64, 8, 8, 256)

    - U-Net
    >>> ....
    >>> conv10 = tl.layers.Conv2dLayer(conv9, act=tf.nn.relu,
    ...        shape=[3,3,1024,1024], strides=[1,1,1,1], padding='SAME',
    ...        W_init=w_init, b_init=b_init, name='conv10')
    >>> print(conv10.outputs)
    ... (batch_size, 32, 32, 1024)
    >>> deconv1 = tl.layers.DeConv2dLayer(conv10, act=tf.nn.relu,
    ...         shape=[3,3,512,1024], strides=[1,2,2,1], output_shape=[batch_size,64,64,512],
    ...         padding='SAME', W_init=w_init, b_init=b_init, name='devcon1_1')
    """
    def __init__(
        self,
        layer = None,
        act = tf.identity,
        shape = [3, 3, 128, 256],
        output_shape = [1, 256, 256, 128],
        strides = [1, 2, 2, 1],
        padding = 'SAME',
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='decnn2d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate DeConv2dLayer %s: %s, %s, %s, %s, %s" %
                            (self.name, str(shape), str(output_shape), str(strides), padding, act.__name__))
        # print("  DeConv2dLayer: Untested")
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_deconv2d', shape=shape, initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b_deconv2d', shape=(shape[-2]), initializer=b_init, **b_init_args )
                self.outputs = act( tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding) + b )
            else:
                self.outputs = act( tf.nn.conv2d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )

class Conv3dLayer(Layer):
    """
    The :class:`Conv3dLayer` class is a 3D CNN layer, see `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_depth, filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints. 1-D of length 4.
        The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer
        The initializer for initializing the bias vector.
    W_init_args : dictionary
        The arguments for the weights initializer.
    b_init_args : dictionary
        The arguments for the biases initializer.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        act = tf.identity,
        shape = [2, 2, 2, 64, 128],
        strides=[1, 2, 2, 2, 1],
        padding='SAME',
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='cnn3d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate Conv3dLayer %s: %s, %s, %s, %s" % (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            # W = tf.Variable(W_init(shape=shape, **W_init_args), name='W_conv')
            # b = tf.Variable(b_init(shape=[shape[-1]], **b_init_args), name='b_conv')
            W = tf.get_variable(name='W_conv3d', shape=shape, initializer=W_init, **W_init_args )
            b = tf.get_variable(name='b_conv3d', shape=(shape[-1]), initializer=b_init, **b_init_args )
            self.outputs = act( tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, name=None) + b )

        # self.outputs = act( tf.nn.conv3d(self.inputs, W, strides=strides, padding=padding, name=None) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

class DeConv3dLayer(Layer):
    """The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [depth, height, width, output_channels, in_channels], filter's in_channels dimension must match that of value.
    output_shape : list of output shape
        representing the output shape of the deconvolution op.
    strides : a list of ints.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer
        The initializer for initializing the bias vector.
    W_init_args : dictionary
        The arguments for the weights initializer.
    b_init_args : dictionary
        The arguments for the biases initializer.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        act = tf.identity,
        shape = [2, 2, 2, 128, 256],
        output_shape = [1, 12, 32, 32, 128],
        strides = [1, 2, 2, 2, 1],
        padding = 'SAME',
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='decnn3d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate DeConv3dLayer %s: %s, %s, %s, %s, %s" %
                            (self.name, str(shape), str(output_shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_deconv3d', shape=shape, initializer=W_init, **W_init_args )
            b = tf.get_variable(name='b_deconv3d', shape=(shape[-2]), initializer=b_init, **b_init_args )

            self.outputs = act( tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

class UpSampling2dLayer(Layer):
    """The :class:`UpSampling2dLayer` class is upSampling 2d layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`_.

    Parameters
    -----------
    layer : a layer class with 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    size : a tupe of int or float.
        (height, width) scale factor or new size of height and width.
    is_scale : boolean, if True (default), size is scale factor, otherwise, size is number of pixels of height and width.
    method : 0, 1, 2, 3. ResizeMethod. Defaults to ResizeMethod.BILINEAR.
        - ResizeMethod.BILINEAR, Bilinear interpolation.
        - ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
        - ResizeMethod.BICUBIC, Bicubic interpolation.
        - ResizeMethod.AREA, Area interpolation.
    align_corners : bool. If true, exactly align all 4 corners of the input and output. Defaults to false.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        size = [],
        is_scale = True,
        method = 0,
        align_corners = False,
        name ='upsample2d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [size_h, size_w]
        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [size_h, size_w]
        else:
            raise Exception("Donot support shape %s" % self.inputs.get_shape())
        print("  tensorlayer:Instantiate UpSampling2dLayer %s: is_scale:%s : %s, method: %d, align_corners: %s" %
                                (name, is_scale, size, method, align_corners))
        with tf.variable_scope(name) as vs:
            try:
                self.outputs = tf.image.resize_images(self.inputs, size=size, method=method, align_corners=align_corners)
            except: # for TF 0.10
                self.outputs = tf.image.resize_images(self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


class DownSampling2dLayer(Layer):
    """The :class:`DownSampling2dLayer` class is downSampling 2d layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`_.

    Parameters
    -----------
    layer : a layer class with 4-D Tensor of shape [batch, height, width, channels] or 3-D Tensor of shape [height, width, channels].
    size : a tupe of int or float.
        (height, width) scale factor or new size of height and width.
    is_scale : boolean, if True (default), size is scale factor, otherwise, size is number of pixels of height and width.
    method : 0, 1, 2, 3. ResizeMethod. Defaults to ResizeMethod.BILINEAR.
        - ResizeMethod.BILINEAR, Bilinear interpolation.
        - ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
        - ResizeMethod.BICUBIC, Bicubic interpolation.
        - ResizeMethod.AREA, Area interpolation.
    align_corners : bool. If true, exactly align all 4 corners of the input and output. Defaults to false.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        size = [],
        is_scale = True,
        method = 0,
        align_corners = False,
        name ='downsample2d_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [size_h, size_w]
        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [size_h, size_w]
        else:
            raise Exception("Donot support shape %s" % self.inputs.get_shape())
        print("  tensorlayer:Instantiate DownSampling2dLayer %s: is_scale:%s : %s, method: %d, align_corners: %s" %
                                (name, is_scale, size, method, align_corners))
        with tf.variable_scope(name) as vs:
            try:
                self.outputs = tf.image.resize_images(self.inputs, size=size, method=method, align_corners=align_corners)
            except: # for TF 0.10
                self.outputs = tf.image.resize_images(self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


class AtrousConv2dLayer(Layer):
    """The :class:`AtrousConv2dLayer` class is Atrous convolution (a.k.a. convolution with holes or dilated convolution) 2D layer, see `tf.nn.atrous_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#atrous_conv2d>`_.

    Parameters
    -----------
    layer: a layer class with 4-D Tensor of shape [batch, height, width, channels].
    # filters : A 4-D Tensor with the same type as value and shape [filter_height, filter_width, in_channels, out_channels]. filters' in_channels dimension must match that of value. Atrous convolution is equivalent to standard convolution with upsampled filters with effective height filter_height + (filter_height - 1) * (rate - 1) and effective width filter_width + (filter_width - 1) * (rate - 1), produced by inserting rate - 1 zeros along consecutive elements across the filters' spatial dimensions.
    n_filter : number of filter.
    filter_size : tuple (height, width) for filter size.
    rate : A positive int32. The stride with which we sample input values across the height and width dimensions. Equivalently, the rate by which we upsample the filter values by inserting zeros across the height and width dimensions. In the literature, the same parameter is sometimes called input stride or dilation.
    act : activation function, None for linear.
    padding : A string, either 'VALID' or 'SAME'. The padding algorithm.
    W_init : weights initializer. The initializer for initializing the weight matrix.
    b_init : biases initializer or None. The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary. The arguments for the weights tf.get_variable().
    b_init_args : dictionary. The arguments for the biases tf.get_variable().
    name : a string or None, an optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        n_filter = 32,
        filter_size = (3,3),
        rate = 2,
        act = None,
        padding = 'SAME',
        W_init = tf.truncated_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name = 'atrou2d'
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate AtrousConv2dLayer %s: n_filter: %d, filter_size: %s, rate: %d, padding: %s, act: %s" %
                            (self.name, n_filter, filter_size, rate, padding, act.__name__))
        if act is None:
            act = tf.identity
        with tf.variable_scope(name) as vs:
            shape = [filter_size[0], filter_size[1], int(self.inputs.get_shape()[-1]), n_filter]
            filters = tf.get_variable(name='filter', shape=shape, initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b', shape=(n_filter), initializer=b_init, **b_init_args )
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding) + b)
            else:
                self.outputs = act(tf.nn.atrous_conv2d(self.inputs, filters, rate, padding))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init:
            self.all_params.extend( [filters, b] )
        else:
            self.all_params.extend( [filters] )

class SeparableConv2dLayer(Layer):#TODO
    """The :class:`SeparableConv2dLayer` class is 2-D convolution with separable filters., see `tf.nn.separable_conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#separable_conv2d>`_.

    Parameters
    -----------
    layer: a layer class with 4-D Tensor of shape [batch, height, width, channels].
    depthwise_filter : 4-D Tensor with shape [filter_height, filter_width, in_channels, channel_multiplier]. Contains in_channels convolutional filters of depth 1.
    pointwise_filter : 4-D Tensor with shape [1, 1, channel_multiplier * in_channels, out_channels]. Pointwise filter to mix channels after depthwise_filter has convolved spatially.
    strides : 1-D of size 4. The strides for the depthwise convolution for each dimension of input.
    padding : A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
    name : a string or None, an optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        depthwise_filter = None,
        pointwise_filter = None,
        rate = 2,
        padding = 'SAME',
        name = 'atrou2d'
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # print("  tensorlayer:Instantiate SeparableConv2dLayer %s: %s, %s, %s, %s" %
        #                     (self.name, str(shape), str(strides), padding, act.__name__))
        # with tf.variable_scope(name) as vs:
        #     self.outputs = tf.nn.separable_conv2d(value, filters, rate, padding)
        #
        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        # self.all_layers.extend( [self.outputs] )

## Initializers for Convuolutional Layers
def deconv2d_bilinear_upsampling_initializer(shape):
    """Returns initializer that can be passed to DeConv2dLayer to initalize the
    weights to correspond to channel wise bilinear upsampling.
    Used in some segmantic segmentation approches such as [FCN](https://arxiv.org/abs/1605.06211)

    Parameters
    ----------
        shape : list of shape
            shape of the filters, [height, width, output_channels, in_channels], must match that passed to DeConv2dLayer

    Returns
    ----------
        tf.constant_initializer
            with weights set to correspond to per channel bilinear upsampling when passed as W_int in DeConv2dLayer

    Examples
    --------
    >>> rescale_factor = 2 #upsampling by a factor of 2, ie e.g 100->200
    >>> filter_size = (2 * rescale_factor - rescale_factor % 2) #Corresponding bilinear filter size
    >>> num_in_channels = 3
    >>> num_out_channels = 3
    >>> deconv_filter_shape = [filter_size, filter_size, num_out_channels, num_in_channels]
    >>> x = tf.placeholder(tf.float32, [1, imsize, imsize, num_channels])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> bilinear_init = deconv2d_bilinear_upsampling_initializer(shape=filter_shape)
    >>> network = tl.layers.DeConv2dLayer(network,
                            shape = filter_shape,
                            output_shape = [1, imsize*rescale_factor, imsize*rescale_factor, num_out_channels],
                            strides=[1, rescale_factor, rescale_factor, 1],
                            W_init=bilinear_init,
                            padding='SAME',
                            act=tf.identity, name='g/h1/decon2d')
    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')
    if shape[3] < shape [2]:
        raise Exception('deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels ')

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    #Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels))
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_weights_init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return bilinear_weights_init

## Convolutional layer (Simplified)
def Conv2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act = None,
        padding='SAME', W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
        W_init_args = {}, b_init_args = {}, name ='conv2d',):
    """Wrapper for :class:`Conv2dLayer`, if you don't understand how to use :class:`Conv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : number of filter.
    filter_size : tuple (height, width) for filter size.
    strides : tuple (height, width) for strides.
    act : None or activation function.
    others : see :class:`Conv2dLayer`.

    Examples
    --------
    >>> w_init = tf.truncated_normal_initializer(stddev=0.01)
    >>> b_init = tf.constant_initializer(value=0.0)
    >>> inputs = InputLayer(x, name='inputs')
    >>> conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    >>> conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    >>> pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
    >>> conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    >>> conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    >>> pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
    """
    # assert len(strides) == 2, print("len(strides)==2, Conv2d and Conv2dLayer are different.")
    if act is None:
        act = tf.identity
    net = Conv2dLayer(net,
                       act = act,
                       shape = [filter_size[0], filter_size[1], int(net.outputs.get_shape()[-1]), n_filter],  # 32 features for each 5x5 patch
                       strides = [1, strides[0], strides[1], 1],
                       padding = padding,
                       W_init = W_init,
                       W_init_args = W_init_args,
                       b_init = b_init,
                       b_init_args = b_init_args,
                       name = name)
    return net

def DeConv2d(net, n_out_channel = 32, filter_size=(3, 3),
        out_size = (30, 30), strides = (2, 2), padding = 'SAME', batch_size = None, act = None,
        W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
        W_init_args = {}, b_init_args = {}, name ='decnn2d'):
    """Wrapper for :class:`DeConv2dLayer`, if you don't understand how to use :class:`DeConv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_out_channel : int, number of output channel.
    filter_size : tuple of (height, width) for filter size.
    out_size :  tuple of (height, width) of output.
    batch_size : int or None, batch_size. If None, try to find the batch_size from the first dim of net.outputs (you should tell the batch_size when define the input placeholder).
    strides : tuple of (height, width) for strides.
    act : None or activation function.
    others : see :class:`DeConv2dLayer`.
    """
    # assert len(strides) == 2, print("len(strides)==2, DeConv2d and DeConv2dLayer are different.")
    if act is None:
        act = tf.identity
    if batch_size is None:
        batch_size = tf.shape(net.outputs)[0]
    net = DeConv2dLayer(layer = net,
                    act = act,
                    shape = [filter_size[0], filter_size[1], n_out_channel, int(net.outputs.get_shape()[-1])],
                    output_shape = [batch_size, int(out_size[0]), int(out_size[1]), n_out_channel],
                    strides = [1, strides[0], strides[1], 1],
                    padding = padding,
                    W_init = W_init,
                    b_init = b_init,
                    W_init_args = W_init_args,
                    b_init_args = b_init_args,
                    name = name)
    return net

def MaxPool2d(net, filter_size=(2, 2), strides=None, padding='SAME', name='maxpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : TensorLayer layer.
    filter_size : tuple of (height, width) for filter size.
    strides : tuple of (height, width). Default is the same with filter_size.
    others : see :class:`PoolLayer`.
    """
    if strides is None:
        strides = filter_size
    # assert len(strides) == 2, print("len(strides)==2, MaxPool2d and PoolLayer are different.")
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            padding=padding,
            pool = tf.nn.max_pool,
            name = name)
    return net

def MeanPool2d(net, filter_size=(2, 2), strides=None, padding='SAME', name='meanpool'):
    """Wrapper for :class:`PoolLayer`.

    Parameters
    -----------
    net : TensorLayer layer.
    filter_size : tuple of (height, width) for filter size.
    strides : tuple of (height, width). Default is the same with filter_size.
    others : see :class:`PoolLayer`.
    """
    if strides is None:
        strides = filter_size
    # assert len(strides) == 2, print("len(strides)==2, MeanPool2d and PoolLayer are different.")
    net = PoolLayer(net, ksize=[1, filter_size[0], filter_size[1], 1],
            strides=[1, strides[0], strides[1], 1],
            padding=padding,
            pool = tf.nn.avg_pool,
            name = name)
    return net

# ## Normalization layer
class LocalResponseNormLayer(Layer):
    """The :class:`LocalResponseNormLayer` class is for Local Response Normalization, see ``tf.nn.local_response_normalization``.
    The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.
    Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius.

    Parameters
    -----------
    layer : a layer class. Must be one of the following types: float32, half. 4-D.
    depth_radius : An optional int. Defaults to 5. 0-D. Half-width of the 1-D normalization window.
    bias : An optional float. Defaults to 1. An offset (usually positive to avoid dividing by 0).
    alpha : An optional float. Defaults to 1. A scale factor, usually positive.
    beta : An optional float. Defaults to 0.5. An exponent.
    name : A string or None, an optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        depth_radius = None,
        bias = None,
        alpha = None,
        beta = None,
        name ='lrn_layer',
    ):
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate LocalResponseNormLayer %s: depth_radius: %d, bias: %f, alpha: %f, beta: %f" %
                            (self.name, depth_radius, bias, alpha, beta))
        with tf.variable_scope(name) as vs:
            self.outputs = tf.nn.local_response_normalization(self.inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.zeros_initializer,
        # gamma_init = tf.ones_initializer,
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
        name ='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
                            (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        # def _get_variable(name,
        #                   shape,
        #                   initializer,
        #                   weight_decay=0.0,
        #                   dtype='float',
        #                   trainable=True):
        #     "A little wrapper around tf.get_variable to do weight decay and add to"
        #     "resnet collection"
        #     if weight_decay > 0:
        #         regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        #     else:
        #         regularizer = None
        #     # collections = [TF_GRAPHKEYS_VARIABLES, RESNET_VARIABLES]
        #     return tf.get_variable(name,
        #                            shape=shape,
        #                            initializer=initializer,
        #                            dtype=dtype,
        #                            regularizer=regularizer,
        #                         #    collections=collections,
        #                            trainable=trainable)

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            # beta = _get_variable('beta',
            #                      params_shape,
            #                      initializer=beta_init)
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)#, restore=restore)
            # try: # TF12
            #     gamma = _get_variable('gamma',
            #                           params_shape,
            #                           initializer=gamma_init())
            # except: # TF11
            # gamma = _get_variable('gamma',
            #                       params_shape,
            #                       initializer=gamma_init)
            # print("x"*100)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )#restore=restore)

            ## 2. moving variables during training (not update by gradient!)
            # trainable=False means : it prevent TF from updating this variable
            # from the gradient, we have to update this from the mean computed
            # from each batch during training
            # moving_mean = _get_variable('moving_mean',
            #                             params_shape,
            #                             initializer=tf.zeros_initializer,
            #                             trainable=False)
            # try: # TF12
            #     moving_variance = _get_variable('moving_variance',
            #                                     params_shape,
            #                                     initializer=tf.ones_initializer(),
            #                                     trainable=False)
            # except: # TF11
            #     moving_variance = _get_variable('moving_variance',
            #                                     params_shape,
            #                                     initializer=tf.ones_initializer,
            #                                     trainable=False)

            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False,)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False,)#   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            # ema = tf.train.ExponentialMovingAverage(decay=decay)    # Akara
            # def mean_var_with_update():
            #     ema_apply_op = ema.apply([moving_mean, moving_variance])
            #     with tf.control_dependencies([ema_apply_op]):
            #         return tf.identity(mean), tf.identity(variance)

            ## 4. behaviour for training and testing
            # if not is_train:    # test : mean=0, std=1
            # # if is_train:      # train : mean=0, std=1
            #     is_train = tf.cast(tf.ones([]), tf.bool)
            # else:
            #     is_train = tf.cast(tf.zeros([]), tf.bool)
            #
            # # mean, var = control_flow_ops.cond(
            # mean, var = tf.cond(
            #     # is_train, lambda: (mean, variance),     # when training, (x-mean(x))/var(x)
            #     is_train, mean_var_with_update,
            #     lambda: (moving_mean, moving_variance)) # when inferencing, (x-0)/1
            #
            # self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            else:
                # self.outputs = act( tf.nn.batch_normalization(self.inputs, ema.average(mean), ema.average(variance), beta, gamma, epsilon) ) # Akara
                self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )    # Simiao
                # self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, variance, beta, gamma, epsilon) )

            # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)  # 8 params in TF12 if zero_debias=True
            # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
            variables = [beta, gamma, moving_mean, moving_variance]
            # variables = [beta, gamma]

            # print(len(variables))
            # for idx, v in enumerate(variables):
            #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )
        # self.all_params.extend( [beta, gamma] )



class BatchNormLayer_TF(Layer):   # Work well TF contrib https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float
        A decay factor for ExponentialMovingAverage.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """
    def __init__(
        self,
        layer = None,
        decay = 0.95,#.999,
        center = True,
        scale = True,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.zeros_initializer,
        # gamma_init = tf.ones_initializer,
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
        name ='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
                            (self.name, decay, epsilon, act.__name__, is_train))
        from tensorflow.contrib.layers.python.layers import utils
        from tensorflow.contrib.framework.python.ops import variables
        from tensorflow.python.ops import init_ops
        from tensorflow.python.ops import nn
        from tensorflow.python.training import moving_averages
        from tensorflow.python.framework import ops
        from tensorflow.python.ops import variable_scope
        variables_collections = None
        outputs_collections=None
        updates_collections=None#ops.GraphKeys.UPDATE_OPS
        # with variable_scope.variable_op_scope([inputs],
        #                                     scope, 'BatchNorm', reuse=reuse) as sc:
        # with variable_scope.variable_op_scope([self.inputs], None, name) as vs:
        with tf.variable_scope(name) as vs:
            inputs_shape = self.inputs.get_shape()
            dtype = self.inputs.dtype.base_dtype
            axis = list(range(len(inputs_shape) - 1)) # [0, 1, 2]
            params_shape = inputs_shape[-1:]
            # Allocate parameters for the beta and gamma of the normalization.
            beta, gamma = None, None
            if center:
              beta_collections = utils.get_variable_collections(variables_collections,
                                                                'beta')
              beta = variables.model_variable('beta',
                                              shape=params_shape,
                                              dtype=dtype,
                                            #   initializer=init_ops.zeros_initializer,
                                              initializer=beta_init,
                                              collections=beta_collections,)
                                            #   trainable=trainable)
            if scale:
              gamma_collections = utils.get_variable_collections(variables_collections,
                                                                 'gamma')
              gamma = variables.model_variable('gamma',
                                               shape=params_shape,
                                               dtype=dtype,
                                            #    initializer=init_ops.ones_initializer,
                                               initializer=gamma_init,
                                               collections=gamma_collections,)
                                            #    trainable=trainable)
            # Create moving_mean and moving_variance variables and add them to the
            # appropiate collections.
            moving_mean_collections = utils.get_variable_collections(
                variables_collections,
                'moving_mean')
            moving_mean = variables.model_variable(
                'moving_mean',
                shape=params_shape,
                dtype=dtype,
                # initializer=init_ops.zeros_initializer,
                initializer=tf.zeros_initializer,
                trainable=False,
                collections=moving_mean_collections)
            moving_variance_collections = utils.get_variable_collections(
                variables_collections,
                'moving_variance')
            moving_variance = variables.model_variable(
                'moving_variance',
                shape=params_shape,
                dtype=dtype,
                # initializer=init_ops.ones_initializer,
                initializer=tf.constant_initializer(1.),
                trainable=False,
                collections=moving_variance_collections)
            if is_train:
              # Calculate the moments based on the individual batch.
              mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
              # Update the moving_mean and moving_variance moments.
            #   update_moving_mean = moving_averages.assign_moving_average(
            #       moving_mean, mean, decay)
            #   update_moving_variance = moving_averages.assign_moving_average(
            #       moving_variance, variance, decay)
            #   if updates_collections is None:
            #     # Make sure the updates are computed here.
            #       with ops.control_dependencies([update_moving_mean,
            #                                        update_moving_variance]):
            #          outputs = nn.batch_normalization(
            #               self.inputs, mean, variance, beta, gamma, epsilon)

              update_moving_mean = tf.assign(moving_mean,
                                   moving_mean * decay + mean * (1 - decay))
              update_moving_variance = tf.assign(moving_variance,
                                  moving_variance * decay + variance * (1 - decay))
              with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                  outputs = nn.batch_normalization(
                              self.inputs, mean, variance, beta, gamma, epsilon)
            #   else:
            #     # Collect the updates to be computed later.
            #     ops.add_to_collections(updates_collections, update_moving_mean)
            #     ops.add_to_collections(updates_collections, update_moving_variance)
            #     outputs = nn.batch_normalization(
            #         self.inputs, mean, variance, beta, gamma, epsilon)
            else:
            #   mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
              outputs = nn.batch_normalization(
                  self.inputs, moving_mean, moving_variance, beta, gamma, epsilon)
                # self.inputs, mean, variance, beta, gamma, epsilon)
            outputs.set_shape(self.inputs.get_shape())
            # if activation_fn:
            self.outputs = act(outputs)

            # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
            # return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
            variables = [beta, gamma, moving_mean, moving_variance]

        mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
        self.check_mean = mean
        self.check_variance = variance

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )

class BatchNormLayer5(Layer):   # Akara Work well
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float
        A decay factor for ExponentialMovingAverage.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.zeros_initializer,
        # gamma_init = tf.ones_initializer,
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
        name ='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
                            (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)#, restore=restore)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )#restore=restore)

            ## 2. moving variables during training (not update by gradient!)
            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False,)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False,)#   restore=restore)

            batch_mean, batch_var = tf.nn.moments(self.inputs, axis)
            ## 3.
            # These ops will only be preformed when training.
            def mean_var_with_update():
                try:    # TF12
                    update_moving_mean = moving_averages.assign_moving_average(
                                    moving_mean, batch_mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                    update_moving_variance = moving_averages.assign_moving_average(
                                    moving_variance, batch_var, decay, zero_debias=False) # if zero_debias=True, has bias
                    # print("TF12 moving")
                except Exception as e:  # TF11
                    update_moving_mean = moving_averages.assign_moving_average(
                                    moving_mean, batch_mean, decay)
                    update_moving_variance = moving_averages.assign_moving_average(
                                    moving_variance, batch_var, decay)
                    # print("TF11 moving")

            # def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    # return tf.identity(update_moving_mean), tf.identity(update_moving_variance)
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # ema = tf.train.ExponentialMovingAverage(decay=decay)    # Akara
            # def mean_var_with_update():
            #     ema_apply_op = ema.apply([batch_mean, batch_var])
            #     with tf.control_dependencies([ema_apply_op]):
            #         return tf.identity(batch_mean), tf.identity(batch_var)

            ## 4. behaviour for training and testing
            # if not is_train:    # test : mean=0, std=1
            # # if is_train:      # train : mean=0, std=1
            #     is_train = tf.cast(tf.ones([]), tf.bool)
            # else:
            #     is_train = tf.cast(tf.zeros([]), tf.bool)
            #
            # # mean, var = control_flow_ops.cond(
            # mean, var = tf.cond(
            #     # is_train, lambda: (mean, variance),     # when training, (x-mean(x))/var(x)
            #     is_train, mean_var_with_update,
            #     lambda: (moving_mean, moving_variance)) # when inferencing, (x-0)/1
            #
            # self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            # if not is_train:
            #     mean, var = mean_var_with_update()
            #     self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            # else:
            #     # self.outputs = act( tf.nn.batch_normalization(self.inputs, ema.average(mean), ema.average(variance), beta, gamma, epsilon) ) # Akara
            #     self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )

            # if not is_train:
            #     is_train = tf.cast(tf.ones([]), tf.bool)
            # else:
            #     is_train = tf.cast(tf.zeros([]), tf.bool)
            #
            # mean, var = tf.cond(
            #   is_train,
            #   mean_var_with_update,
            #   lambda: (moving_mean, moving_variance))

            # if not is_train:
            if is_train:
                mean, var = mean_var_with_update()
                    # mean, var = (update_moving_mean, update_moving_variance)
            else:
                mean, var = (moving_mean, moving_variance)
                # mean, var = (batch_mean, batch_var) # hao

            normed = tf.nn.batch_normalization(
              x=self.inputs,
              mean=mean,
              variance=var,
              offset=beta,
              scale=gamma,
              variance_epsilon=epsilon,
              name="tf_bn"
            )
            self.outputs = act( normed )
            # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)  # 8 params in TF12 if zero_debias=True
            # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
            variables = [beta, gamma, moving_mean, moving_variance]
            # variables = [beta, gamma]

            # print(len(variables))
            # for idx, v in enumerate(variables):
            #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )
        # self.all_params.extend( [beta, gamma] )



# class BatchNormLayer4(Layer): # work TFlearn https://github.com/tflearn/tflearn/blob/master/tflearn/layers/normalization.py
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         input_shape = self.inputs.get_shape()
#         # params_shape = input_shape[-1:]
#         input_ndim = len(input_shape)
#         from tensorflow.python.training import moving_averages
#         from tensorflow.python.ops import control_flow_ops
#
#         # gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)
#
#         # Variable Scope fix for older TF
#         scope = name
#         try:
#             vscope = tf.variable_scope(scope, default_name=name, values=[self.inputs],)
#                                     #    reuse=reuse)
#         except Exception:
#             vscope = tf.variable_op_scope([self.inputs], scope, name)#, reuse=reuse)
#
#         with vscope as scope:
#             name = scope.name
#         # with tf.variable_scope(name) as vs:
#             beta = tf.get_variable('beta', shape=[input_shape[-1]],
#                                 initializer=beta_init,)
#                             #    initializer=tf.constant_initializer(beta),)
#                             #    trainable=trainable, )#restore=restore)
#             gamma = tf.get_variable('gamma', shape=[input_shape[-1]],
#                                 initializer=gamma_init, )#trainable=trainable,)
#                                 # restore=restore)
#
#             axis = list(range(input_ndim - 1))
#             moving_mean = tf.get_variable('moving_mean',
#                                       input_shape[-1:],
#                                       initializer=tf.zeros_initializer,
#                                       trainable=False,)
#                                     #   restore=restore)
#             moving_variance = tf.get_variable('moving_variance',
#                                           input_shape[-1:],
#                                           initializer=tf.constant_initializer(1.),
#                                           trainable=False,)
#                                         #   restore=restore)
#
#             # Define a function to update mean and variance
#             def update_mean_var():
#                 mean, variance = tf.nn.moments(self.inputs, axis)
#
#                 # Fix TF 0.12
#                 try:
#                     update_moving_mean = moving_averages.assign_moving_average(
#                         moving_mean, mean, decay, zero_debias=False)            # if zero_debias=True, accuracy is high ..
#                     update_moving_variance = moving_averages.assign_moving_average(
#                         moving_variance, variance, decay, zero_debias=False)
#                 except Exception as e:  # TF 11
#                     update_moving_mean = moving_averages.assign_moving_average(
#                         moving_mean, mean, decay)
#                     update_moving_variance = moving_averages.assign_moving_average(
#                         moving_variance, variance, decay)
#
#                 with tf.control_dependencies(
#                         [update_moving_mean, update_moving_variance]):
#                     return tf.identity(mean), tf.identity(variance)
#
#             # Retrieve variable managing training mode
#             # is_training = tflearn.get_training_mode()
#             if not is_train:    # test : mean=0, std=1
#             # if is_train:      # train : mean=0, std=1
#                 is_training = tf.cast(tf.ones([]), tf.bool)
#             else:
#                 is_training = tf.cast(tf.zeros([]), tf.bool)
#             mean, var = tf.cond(
#                 is_training, update_mean_var, lambda: (moving_mean, moving_variance))
#                             #  ones                 zeros
#             try:
#                 inference = tf.nn.batch_normalization(
#                     self.inputs, mean, var, beta, gamma, epsilon)
#                 inference.set_shape(input_shape)
#             # Fix for old Tensorflow
#             except Exception as e:
#                 inference = tf.nn.batch_norm_with_global_normalization(
#                     self.inputs, mean, var, beta, gamma, epsilon,
#                     scale_after_normalization=True,
#                 )
#                 inference.set_shape(input_shape)
#
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         # Add attributes for easy access
#         # inference.scope = scope
#         inference.scope = name
#         inference.beta = beta
#         inference.gamma = gamma
#
#         self.outputs = act( inference )
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )

# class BatchNormLayer2(Layer):   # don't work http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         x_shape = self.inputs.get_shape()
#         params_shape = x_shape[-1:]
#
#         with tf.variable_scope(name) as vs:
#             gamma = tf.get_variable("gamma", shape=params_shape,
#                         initializer=gamma_init)
#             beta = tf.get_variable("beta", shape=params_shape,
#                         initializer=beta_init)
#             pop_mean = tf.get_variable("pop_mean", shape=params_shape,
#                         initializer=tf.zeros_initializer, trainable=False)
#             pop_var = tf.get_variable("pop_var", shape=params_shape,
#                         initializer=tf.constant_initializer(1.), trainable=False)
#
#             if is_train:
#                 batch_mean, batch_var = tf.nn.moments(self.inputs, list(range(len(x_shape) - 1)))
#                 train_mean = tf.assign(pop_mean,
#                                        pop_mean * decay + batch_mean * (1 - decay))
#                 train_var = tf.assign(pop_var,
#                                       pop_var * decay + batch_var * (1 - decay))
#                 with tf.control_dependencies([train_mean, train_var]):
#                     self.outputs = act(tf.nn.batch_normalization(self.inputs,
#                         batch_mean, batch_var, beta, gamma, epsilon))
#             else:
#                 self.outputs = act(tf.nn.batch_normalization(self.inputs,
#                     pop_mean, pop_var, beta, gamma, epsilon))
#                     # self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, variance, beta, gamma, epsilon) )
#             # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)  # 8 params in TF12 if zero_debias=True
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )


# class BatchNormLayer3(Layer):   # don't work http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         """
#         Batch normalization on convolutional maps.
#         Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#         Args:
#             x:           Tensor, 4D BHWD input maps
#             n_out:       integer, depth of input maps
#             phase_train: boolean tf.Varialbe, true indicates training phase
#             scope:       string, variable scope
#         Return:
#             normed:      batch-normalized maps
#         """
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         x_shape = self.inputs.get_shape()
#         params_shape = x_shape[-1:]
#
#         if is_train:
#             phase_train = tf.cast(tf.ones([]), tf.bool)
#         else:
#             phase_train = tf.cast(tf.zeros([]), tf.bool)
#
#         with tf.variable_scope(name) as vs:
#             gamma = tf.get_variable("gamma", shape=params_shape,
#                         initializer=gamma_init)
#             beta = tf.get_variable("beta", shape=params_shape,
#                         initializer=beta_init)
#             batch_mean, batch_var = tf.nn.moments(self.inputs, list(range(len(x_shape) - 1)),#[0,1,2],
#                             name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([batch_mean, batch_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#
#             mean, var = tf.cond(phase_train,
#                                 mean_var_with_update,
#                                 lambda: (ema.average(batch_mean), ema.average(batch_var)))
#             normed = tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon)
#             self.outputs = act( normed )
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )


# class BatchNormLayer_old(Layer):  # don't work
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     is_train : boolean
#         Whether train or inference.
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `tf.nn.batch_normalization <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.batch_normalization.md>`_
#     - `stackoverflow <http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow>`_
#     - `tensorflow.contrib <https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         act = tf.identity,
#         decay = 0.999,
#         epsilon = 0.001,
#         is_train = None,
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, is_train: %s" %
#                             (self.name, decay, epsilon, is_train))
#         if is_train == None:
#             raise Exception("is_train must be True or False")
#
#         # (name, input_var, decay, epsilon, is_train)
#         inputs_shape = self.inputs.get_shape()
#         axis = list(range(len(inputs_shape) - 1))
#         params_shape = inputs_shape[-1:]
#
#         with tf.variable_scope(name) as vs:
#             beta = tf.get_variable(name='beta', shape=params_shape,
#                                  initializer=tf.constant_initializer(0.0))
#             gamma = tf.get_variable(name='gamma', shape=params_shape,
#                                   initializer=tf.constant_initializer(1.0))
#             batch_mean, batch_var = tf.nn.moments(self.inputs,
#                                                 axis,
#                                                 name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#             def mean_var_with_update():
#               ema_apply_op = ema.apply([batch_mean, batch_var])
#               with tf.control_dependencies([ema_apply_op]):
#                   return tf.identity(batch_mean), tf.identity(batch_var)
#
#             if is_train:
#                 is_train = tf.cast(tf.ones(1), tf.bool)
#             else:
#                 is_train = tf.cast(tf.zeros(1), tf.bool)
#
#             is_train = tf.reshape(is_train, [])
#
#             # print(is_train)
#             # exit()
#
#             mean, var = tf.cond(
#               is_train,
#               mean_var_with_update,
#               lambda: (ema.average(batch_mean), ema.average(batch_var))
#             )
#             normed = tf.nn.batch_normalization(
#               x=self.inputs,
#               mean=mean,
#               variance=var,
#               offset=beta,
#               scale=gamma,
#               variance_epsilon=epsilon,
#               name='tf_bn'
#             )
#         self.outputs = act( normed )
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( [beta, gamma] )


## Pooling layer
class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a Pooling layer, you can choose
    ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D or
    ``tf.nn.max_pool3d()`` and ``tf.nn.avg_pool3d()`` for 3D.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    ksize : a list of ints that has length >= 4.
        The size of the window for each dimension of the input tensor.
    strides : a list of ints that has length >= 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    pool : a pooling function
        - see `TensorFlow pooling APIs <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`_
        - class ``tf.nn.max_pool``
        - class ``tf.nn.avg_pool``
        - class ``tf.nn.max_pool3d``
        - class ``tf.nn.avg_pool3d``
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - see :class:`Conv2dLayer`.
    """
    def __init__(
        self,
        layer = None,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        pool = tf.nn.max_pool,
        name ='pool_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate PoolLayer   %s: %s, %s, %s, %s" %
                            (self.name, str(ksize), str(strides), padding, pool.__name__))

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


## Recurrent layer
class RNNLayer(Layer):
    """
    The :class:`RNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    return_last : boolean
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    Examples
    --------
    - For words
    >>> input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    >>> network = tl.layers.EmbeddingInputlayer(
    ...                 inputs = input_data,
    ...                 vocabulary_size = vocab_size,
    ...                 embedding_size = hidden_size,
    ...                 E_init = tf.random_uniform_initializer(-init_scale, init_scale),
    ...                 name ='embedding_layer')
    >>> if is_training:
    >>>     network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop1')
    >>> network = tl.layers.RNNLayer(network,
    ...             cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
    ...             cell_init_args={'forget_bias': 0.0},# 'state_is_tuple': True},
    ...             n_hidden=hidden_size,
    ...             initializer=tf.random_uniform_initializer(-init_scale, init_scale),
    ...             n_steps=num_steps,
    ...             return_last=False,
    ...             name='basic_lstm_layer1')
    >>> lstm1 = network
    >>> if is_training:
    >>>     network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop2')
    >>> network = tl.layers.RNNLayer(network,
    ...             cell_fn=tf.nn.rnn_cell.BasicLSTMCell,
    ...             cell_init_args={'forget_bias': 0.0}, # 'state_is_tuple': True},
    ...             n_hidden=hidden_size,
    ...             initializer=tf.random_uniform_initializer(-init_scale, init_scale),
    ...             n_steps=num_steps,
    ...             return_last=False,
    ...             return_seq_2d=True,
    ...             name='basic_lstm_layer2')
    >>> lstm2 = network
    >>> if is_training:
    >>>     network = tl.layers.DropoutLayer(network, keep=keep_prob, name='drop3')
    >>> network = tl.layers.DenseLayer(network,
    ...             n_units=vocab_size,
    ...             W_init=tf.random_uniform_initializer(-init_scale, init_scale),
    ...             b_init=tf.random_uniform_initializer(-init_scale, init_scale),
    ...             act = tl.activation.identity, name='output_layer')

    - For CNN+LSTM
    >>> x = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                         act = tf.nn.relu,
    ...                         shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    ...                         strides=[1, 2, 2, 1],
    ...                         padding='SAME',
    ...                         name ='cnn_layer1')
    >>> network = tl.layers.PoolLayer(network,
    ...                         ksize=[1, 2, 2, 1],
    ...                         strides=[1, 2, 2, 1],
    ...                         padding='SAME',
    ...                         pool = tf.nn.max_pool,
    ...                         name ='pool_layer1')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                         act = tf.nn.relu,
    ...                         shape = [5, 5, 32, 10], # 10 features for each 5x5 patch
    ...                         strides=[1, 2, 2, 1],
    ...                         padding='SAME',
    ...                         name ='cnn_layer2')
    >>> network = tl.layers.PoolLayer(network,
    ...                         ksize=[1, 2, 2, 1],
    ...                         strides=[1, 2, 2, 1],
    ...                         padding='SAME',
    ...                         pool = tf.nn.max_pool,
    ...                         name ='pool_layer2')
    >>> network = tl.layers.FlattenLayer(network, name='flatten_layer')
    >>> network = tl.layers.ReshapeLayer(network, shape=[-1, num_steps, int(network.outputs._shape[-1])])
    >>> rnn1 = tl.layers.RNNLayer(network,
    ...                         cell_fn=tf.nn.rnn_cell.LSTMCell,
    ...                         cell_init_args={},
    ...                         n_hidden=200,
    ...                         initializer=tf.random_uniform_initializer(-0.1, 0.1),
    ...                         n_steps=num_steps,
    ...                         return_last=False,
    ...                         return_seq_2d=True,
    ...                         name='rnn_layer')
    >>> network = tl.layers.DenseLayer(rnn1, n_units=3,
    ...                         act = tl.activation.identity, name='output_layer')

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`ReshapeLayer`.

    References
    ----------
    - `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_
    - `tensorflow/python/ops/rnn.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py>`_
    - `tensorflow/python/ops/rnn_cell.py <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py>`_
    - see TensorFlow tutorial ``ptb_word_lm.py``, TensorLayer tutorials ``tutorial_ptb_lstm*.py`` and ``tutorial_generate_text.py``
    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.BasicRNNCell,
        cell_init_args = {},
        n_hidden = 100,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        n_steps = 5,
        initial_state = None,
        return_last = False,
        # is_reshape = True,
        return_seq_2d = False,
        name = 'rnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate RNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, cell_fn:%s " % (self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")


        # is_reshape : boolean (deprecate)
        #     Reshape the inputs to 3 dimension tensor.\n
        #     If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        #     If input is [batch_size * n_steps, n_features], we need to reshape it.
        # if is_reshape:
        #     self.inputs = tf.reshape(self.inputs, shape=[-1, n_steps, int(self.inputs._shape[-1])])

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        self.cell = cell = cell_fn(num_units=n_hidden, **cell_init_args)
        if initial_state is None:
            self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 1.2.3
        state = self.initial_state
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = outputs[-1]
        else:
            if return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])
            else:
                # <akara>: stack more RNN layer after that
                # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, n_hidden])

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        # print(type(self.outputs))
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )


class BiRNNLayer(Layer):
    """
    The :class:`BiRNNLayer` class is a Bidirectional RNN layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    n_steps : a int
        The sequence length.
    fw_initial_state : None or forward RNN State
        If None, initial_state is zero_state.
    bw_initial_state : None or backward RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_last : boolean
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
        - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    --------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    fw(bw)_final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    fw(bw)_initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    batch_size : int or tensor
        Is int, if able to compute the batch_size, otherwise, tensor for ``?``.

    Notes
    -----
    - Input dimension should be rank 3 : [batch_size, n_steps, n_features], if no, please see :class:`ReshapeLayer`.
    - For predicting, the sequence length has to be the same with the sequence length of training, while, for normal
    RNN, we can use sequence length of 1 for predicting.

    References
    ----------
    - `Source <https://github.com/akaraspt/deepsleep/blob/master/deepsleep/model.py>`_
    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'use_peepholes':True, 'state_is_tuple':True},
        n_hidden = 100,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        n_steps = 5,
        fw_initial_state = None,
        bw_initial_state = None,
        dropout = None,
        n_layer = 1,
        return_last = False,
        return_seq_2d = False,
        name = 'birnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate BiRNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, cell_fn:%s, dropout:%s, n_layer:%d " % (self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            self.batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % self.batch_size)
        else:
            from tensorflow.python.ops import array_ops
            self.batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")

        with tf.variable_scope(name, initializer=initializer) as vs:
            self.fw_cell = cell_fn(num_units=n_hidden, **cell_init_args)
            self.bw_cell = cell_fn(num_units=n_hidden, **cell_init_args)
            # Apply dropout
            if dropout:
                if type(dropout) in [tuple, list]:
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]
                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout
                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                    "float)")
                self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                          self.fw_cell,
                          input_keep_prob=in_keep_prob,
                          output_keep_prob=out_keep_prob)
                self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                          self.bw_cell,
                          input_keep_prob=in_keep_prob,
                          output_keep_prob=out_keep_prob)
            # Apply multiple layers
            if n_layer > 1:
                print("     n_layer: %d" % n_layer)
                try:
                    self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell] * n_layer,
                                                          state_is_tuple=True)
                    self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell] * n_layer,
                                                          state_is_tuple=True)
                except:
                    self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell] * n_layer)
                    self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell] * n_layer)

            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # exit()
            # Feedforward to MultiRNNCell
            list_rnn_inputs = tf.unpack(self.inputs, axis=1)
            outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=list_rnn_inputs,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state
            )

            if return_last:
                self.outputs = outputs[-1]
            else:
                self.outputs = outputs
                if return_seq_2d:
                    # 2D Tensor [n_example, n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden*2])
                else:
                    # <akara>: stack more RNN layer after that
                    # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, n_hidden*2])
            self.fw_final_state = fw_state
            self.bw_final_state = bw_state

            # Retrieve just the RNN variables.
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )

# Advanced Ops for Dynamic RNN
def advanced_indexing_op(input, index):
    """Advanced Indexing for Sequences, returns the outputs by given sequence lengths.
    When return the last output :class:`DynamicRNNLayer` uses it to get the last outputs with the sequence lengths.

    Parameters
    -----------
    input : tensor for data
        [batch_size, n_step(max), n_features]
    index : tensor for indexing, i.e. sequence_length in Dynamic RNN.
        [batch_size]

    Examples
    ---------
    >>> batch_size, max_length, n_features = 3, 5, 2
    >>> z = np.random.uniform(low=-1, high=1, size=[batch_size, max_length, n_features]).astype(np.float32)
    >>> b_z = tf.constant(z)
    >>> sl = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    >>> o = advanced_indexing_op(b_z, sl)
    >>>
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>>
    >>> order = np.asarray([1,1,2])
    >>> print("real",z[0][order[0]-1], z[1][order[1]-1], z[2][order[2]-1])
    >>> y = sess.run([o], feed_dict={sl:order})
    >>> print("given",order)
    >>> print("out", y)
    ... real [-0.93021595  0.53820813] [-0.92548317 -0.77135968] [ 0.89952248  0.19149846]
    ... given [1 1 2]
    ... out [array([[-0.93021595,  0.53820813],
    ...             [-0.92548317, -0.77135968],
    ...             [ 0.89952248,  0.19149846]], dtype=float32)]

    References
    -----------
    - Modified from TFlearn (the original code is used for fixed length rnn), `references <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`_.
    """
    batch_size = tf.shape(input)[0]
    # max_length = int(input.get_shape()[1])    # for fixed length rnn, length is given
    max_length = tf.shape(input)[1]             # for dynamic_rnn, length is unknown
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant

def retrieve_seq_length_op(data):
    """An op to compute the length of a sequence from input shape of [batch_size, n_step(max), n_features],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max), n_features] with zero padding on right hand side.

    Examples
    ---------
    >>> data = [[[1],[2],[0],[0],[0]],
    ...         [[1],[2],[3],[0],[0]],
    ...         [[1],[2],[6],[1],[0]]]
    >>> data = np.asarray(data)
    >>> print(data.shape)
    ... (3, 5, 1)
    >>> data = tf.constant(data)
    >>> sl = retrieve_seq_length_op(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> y = sl.eval()
    ... [2 3 4]

    - Multiple features
    >>> data = [[[1,2],[2,2],[1,2],[1,2],[0,0]],
    ...         [[2,3],[2,4],[3,2],[0,0],[0,0]],
    ...         [[3,3],[2,2],[5,3],[1,2],[0,0]]]
    >>> sl
    ... [4 3 4]

    References
    ------------
    - Borrow from `TFlearn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`_.
    """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length

def retrieve_seq_length_op2(data):
    """An op to compute the length of a sequence, from input shape of [batch_size, n_step(max)],
    it can be used when the features of padding (on right hand side) are all zeros.

    Parameters
    -----------
    data : tensor
        [batch_size, n_step(max)] with zero padding on right hand side.

    Examples
    --------
    >>> data = [[1,2,0,0,0],
    ...         [1,2,3,0,0],
    ...         [1,2,6,1,0]]
    >>> o = retrieve_seq_length_op2(data)
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> print(o.eval())
    ... [2 3 4]
    """
    return tf.reduce_sum(tf.cast(tf.greater(data, tf.zeros_like(data)), tf.int32), 1)


# Dynamic RNN
class DynamicRNNLayer(Layer):
    """
    The :class:`DynamicRNNLayer` class is a Dynamic RNN layer, see ``tf.nn.dynamic_rnn``.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : a tensor, array or None
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence_length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence_length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic_rnn in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_last : boolean
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer or computing cost after it.
        - If False, return 3D Tensor [n_example/n_steps(max), n_steps(max), n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    ------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)

    final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    sequence_length : a tensor or array, shape = [batch_size]
        The sequence lengths computed by Advanced Opt or the given sequence lengths.

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    Examples
    --------
    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input_seqs")
    >>> network = tl.layers.EmbeddingInputlayer(
    ...             inputs = input_seqs,
    ...             vocabulary_size = vocab_size,
    ...             embedding_size = embedding_size,
    ...             name = 'seq_embedding')
    >>> network = tl.layers.DynamicRNNLayer(network,
    ...             cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
    ...             n_hidden = embedding_size,
    ...             dropout = 0.7,
    ...             sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...             return_seq_2d = True,     # stack denselayer or compute cost after it
    ...             name = 'dynamic_rnn')
    ... network = tl.layers.DenseLayer(network, n_units=vocab_size,
    ...             act=tf.identity, name="output")

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_
    - `dynamic_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb>`_
    - `tf.nn.dynamic_rnn <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md>`_
    - `tflearn rnn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`_
    - ``tutorial_dynamic_rnn.py``
    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple' : True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length = None,
        initial_state = None,
        dropout = None,
        n_layer = 1,
        return_last = False,
        return_seq_2d = False,
        name = 'dyrnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate DynamicRNNLayer %s: n_hidden:%d, in_dim:%d %s, cell_fn:%s, dropout:%s, n_layer:%d" % (self.name, n_hidden,
             self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Creats the cell function
        self.cell = cell_fn(num_units=n_hidden, **cell_init_args)

        # Apply dropout
        if dropout:
            if type(dropout) in [tuple, list]:
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                "float)")
            self.cell = tf.nn.rnn_cell.DropoutWrapper(
                      self.cell,
                      input_keep_prob=in_keep_prob,
                      output_keep_prob=out_keep_prob)
        # Apply multiple layers
        if n_layer > 1:
            print("     n_layer: %d" % n_layer)
            try:
                self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * n_layer, state_is_tuple=True)
            except:
                self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * n_layer)

        # Initialize initial_state
        if initial_state is None:
            self.initial_state = self.cell.zero_state(batch_size, dtype=tf.float32)#dtype="float")
        else:
            self.initial_state = initial_state

        # Computes sequence_length
        if sequence_length is None:
            sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))

        # Main - Computes outputs and last_states
        with tf.variable_scope(name, initializer=initializer) as vs:
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=self.cell,
                # inputs=X
                inputs = self.inputs,
                # dtype=tf.float64,
                sequence_length=sequence_length,
                initial_state = self.initial_state,
                )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            print("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            if return_last:
                # [batch_size, n_hidden]
                # outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), n_hidden]
                # self.outputs = result[0]["outputs"]
                # self.outputs = outputs    # it is 3d, but it is a list
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])
                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), n_hidden]
                    max_length = tf.shape(outputs)[1]
                    batch_size = tf.shape(outputs)[0]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [batch_size, max_length, n_hidden])
                    # self.outputs = tf.reshape(tf.concat(1, outputs), [-1, max_length, n_hidden])

        # Final state
        self.final_state = last_states

        self.sequence_length = sequence_length

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )


# Bidirectional Dynamic RNN
class BiDynamicRNNLayer(Layer):
    """
    The :class:`BiDynamicRNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_\n
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : a tensor, array or None
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence_length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence_length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic_rnn in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_.
    fw_initial_state : None or forward RNN State
        If None, initial_state is zero_state.
    bw_initial_state : None or backward RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_last : boolean
        If True, return the last output, "Sequence input and single output"\n
        If False, return all outputs, "Synced sequence input and output"\n
        In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking DenseLayer or computing cost after it.
        - If False, return 3D Tensor [n_example/n_steps(max), n_steps(max), 2 * n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    -----------------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, 2 * n_hidden)

    fw(bw)_final_state : a tensor or StateTuple
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.
        You can get the final state after each iteration during training, then
        feed it to the initial state of next iteration.

    fw(bw)_initial_state : a tensor or StateTuple
        It is the initial state of this RNN layer, you can use it to initialize
        your state at the begining of each epoch or iteration according to your
        training procedure.

    sequence_length : a tensor or array, shape = [batch_size]
        The sequence lengths computed by Advanced Opt or the given sequence lengths.

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.


    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`_
    - `bidirectional_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/bidirectional_rnn.ipynb>`_
    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        sequence_length = None,
        fw_initial_state = None,
        bw_initial_state = None,
        dropout = None,
        n_layer = 1,
        return_last = False,
        return_seq_2d = False,
        name = 'bi_dyrnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate BiDynamicRNNLayer %s: n_hidden:%d, in_dim:%d %s, cell_fn:%s, dropout:%s, n_layer:%d" %
              (self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        with tf.variable_scope(name, initializer=initializer) as vs:
            # Creats the cell function
            self.fw_cell = cell_fn(num_units=n_hidden, **cell_init_args)
            self.bw_cell = cell_fn(num_units=n_hidden, **cell_init_args)

            # Apply dropout
            if dropout:
                if type(dropout) in [tuple, list]:
                    in_keep_prob = dropout[0]
                    out_keep_prob = dropout[1]
                elif isinstance(dropout, float):
                    in_keep_prob, out_keep_prob = dropout, dropout
                else:
                    raise Exception("Invalid dropout type (must be a 2-D tuple of "
                                    "float)")
                self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    self.fw_cell,
                    input_keep_prob=in_keep_prob,
                    output_keep_prob=out_keep_prob)
                self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    self.bw_cell,
                    input_keep_prob=in_keep_prob,
                    output_keep_prob=out_keep_prob)
            # Apply multiple layers
            if n_layer > 1:
                print("     n_layer: %d" % n_layer)
                self.fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.fw_cell] * n_layer)
                self.bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.bw_cell] * n_layer)
            # Initial state of RNN
            if fw_initial_state is None:
                self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.fw_initial_state = fw_initial_state
            if bw_initial_state is None:
                self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, dtype=tf.float32)
            else:
                self.bw_initial_state = bw_initial_state
            # Computes sequence_length
            if sequence_length is None:
                sequence_length = retrieve_seq_length_op(
                    self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))

            outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.fw_cell,
                cell_bw=self.bw_cell,
                inputs=self.inputs,
                sequence_length=sequence_length,
                initial_state_fw=self.fw_initial_state,
                initial_state_bw=self.bw_initial_state,
            )
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            print("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            outputs = tf.concat(2, outputs)
            if return_last:
                # [batch_size, 2 * n_hidden]
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), 2 * n_hidden]
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, 2 * n_hidden]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, 2 * n_hidden])
                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), 2 * n_hidden]
                    max_length = tf.shape(outputs)[1]
                    batch_size = tf.shape(outputs)[0]
                    self.outputs = tf.reshape(tf.concat(1, outputs), [batch_size, max_length, 2 * n_hidden])
                    # self.outputs = tf.reshape(tf.concat(1, outputs), [-1, max_length, 2 * n_hidden])

        # Final state
        self.fw_final_states = states_fw
        self.bw_final_states = states_bw

        self.sequence_length = sequence_length

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )


# Seq2seq
class Seq2Seq(Layer):
    """
    The :class:`Seq2Seq` class is a simple :class:`DynamicRNNLayer` based Seq2seq layer,
    both encoder and decoder are :class:`DynamicRNNLayer`, network details
    see `Model <https://camo.githubusercontent.com/242210d7d0151cae91107ee63bff364a860db5dd/687474703a2f2f6936342e74696e797069632e636f6d2f333031333674652e706e67>`_
    and `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/abs/1409.3215>`_ .

    Parameters
    ----------
    net_encode_in : a :class:`Layer` instance
        Encode sequences, [batch_size, None, n_features].
    net_decode_in : a :class:`Layer` instance
        Decode sequences, [batch_size, None, n_features].
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/rnn_cell/>`_\n
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    encode_sequence_length : tensor for encoder sequence length, see :class:`DynamicRNNLayer` .
    decode_sequence_length : tensor for decoder sequence length, see :class:`DynamicRNNLayer` .
    initial_state : None or forward RNN State
        If None, initial_state is of encoder zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_seq_2d : boolean
        - When return_last = False
        - If True, return 2D Tensor [n_example, 2 * n_hidden], for stacking DenseLayer or computing cost after it.
        - If False, return 3D Tensor [n_example/n_steps(max), n_steps(max), 2 * n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    ------------
    outputs : a tensor
        The output of RNN decoder.

    final_state : a tensor or StateTuple
        Final state of decoder, see :class:`DynamicRNNLayer` .

    Examples
    ----------
    >>> batch_size = 32
    >>> encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    >>> decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    >>> target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    >>> target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()
    >>> with tf.variable_scope("model") as vs:#, reuse=reuse):
    ...     # for chatbot, you can use the same embedding layer,
    ...     # for translation, you may want to use 2 seperated embedding layers
    >>>     net_encode = EmbeddingInputlayer(
    ...             inputs = encode_seqs,
    ...             vocabulary_size = 10000,
    ...             embedding_size = 200,
    ...             name = 'seq_embedding')
    >>>     vs.reuse_variables()
    >>>     tl.layers.set_name_reuse(True)
    >>>     net_decode = EmbeddingInputlayer(
    ...             inputs = decode_seqs,
    ...             vocabulary_size = 10000,
    ...             embedding_size = 200,
    ...             name = 'seq_embedding')
    >>>     net = Seq2Seq(net_encode, net_decode,
    ...             cell_fn = tf.nn.rnn_cell.LSTMCell,
    ...             n_hidden = 200,
    ...             initializer = tf.random_uniform_initializer(-0.1, 0.1),
    ...             encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
    ...             decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
    ...             initial_state = None,
    ...             dropout = None,
    ...             n_layer = 1,
    ...             return_seq_2d = True,
    ...             name = 'seq2seq')
    >>> net_out = DenseLayer(net, n_units=10000, act=tf.identity, name='output')
    >>> e_loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False)
    >>> y = tf.nn.softmax(net_out.outputs)
    >>> net_out.print_params(False)

    Notes
    --------
    - How to feed data: `Sequence to Sequence Learning with Neural Networks <https://arxiv.org/pdf/1409.3215v3.pdf>`_
    - input_seqs : ``['how', 'are', 'you', '<PAD_ID'>]``
    - decode_seqs : ``['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]``
    - target_seqs : ``['I', 'am', 'fine', '<END_ID']``
    - target_mask : ``[1, 1, 1, 1, 0]``
    - related functions : tl.prepro <pad_sequences, precess_sequences, sequences_add_start_id, sequences_get_mask>
    """
    def __init__(
        self,
        net_encode_in = None,
        net_decode_in = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        encode_sequence_length = None,
        decode_sequence_length = None,
        initial_state = None,
        dropout = None,
        n_layer = 1,
        # return_last = False,
        return_seq_2d = False,
        name = 'seq2seq',
    ):
        Layer.__init__(self, name=name)
        # self.inputs = layer.outputs
        print("  tensorlayer:Instantiate Seq2Seq %s: n_hidden:%d, cell_fn:%s, dropout:%s, n_layer:%d" %
              (self.name, n_hidden, cell_fn.__name__, dropout, n_layer))

        with tf.variable_scope(name) as vs:#, reuse=reuse):
            # tl.layers.set_name_reuse(reuse)
            # network = InputLayer(self.inputs, name=name+'/input')
            network_encode = DynamicRNNLayer(net_encode_in,
                     cell_fn = cell_fn,
                     cell_init_args = cell_init_args,
                     n_hidden = n_hidden,
                     initial_state = initial_state,
                     dropout = dropout,
                     n_layer = n_layer,
                     sequence_length = encode_sequence_length,
                     return_last = False,
                     return_seq_2d = True,
                     name = 'encode')
            # vs.reuse_variables()
            # tl.layers.set_name_reuse(True)
            network_decode = DynamicRNNLayer(net_decode_in,
                     cell_fn = cell_fn,
                     cell_init_args = cell_init_args,
                     n_hidden = n_hidden,
                     initial_state = network_encode.final_state,
                     dropout = dropout,
                     n_layer = n_layer,
                     sequence_length = decode_sequence_length,
                     return_last = False,
                     return_seq_2d = return_seq_2d,
                     name = 'decode')
            self.outputs = network_decode.outputs

            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # Final state
        self.final_state = network_decode.final_state

        # self.sequence_length = sequence_length
        self.all_layers = list(network_decode.all_layers)
        self.all_params = list(network_decode.all_params)
        self.all_drop = dict(network_decode.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


class PeekySeq2Seq(Layer):
    """
    Waiting for contribution.
    The :class:`PeekySeq2Seq` class, see `Model <https://camo.githubusercontent.com/7f690d451036938a51e62feb77149c8bb4be6675/687474703a2f2f6936342e74696e797069632e636f6d2f333032617168692e706e67>`_
    and `Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`_ .
    """
    def __init__(
        self,
        net_encode_in = None,
        net_decode_in = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        in_sequence_length = None,
        out_sequence_length = None,
        initial_state = None,
        dropout = None,
        n_layer = 1,
        # return_last = False,
        return_seq_2d = False,
        name = 'peeky_seq2seq',
    ):
        Layer.__init__(self, name=name)
        # self.inputs = layer.outputs
        print("  tensorlayer:Instantiate PeekySeq2seq %s: n_hidden:%d, cell_fn:%s, dropout:%s, n_layer:%d" %
              (self.name, n_hidden, cell_fn.__name__, dropout, n_layer))


class AttentionSeq2Seq(Layer):
    """
    Waiting for contribution.
    The :class:`AttentionSeq2Seq` class, see `Model <https://camo.githubusercontent.com/0e2e4e5fb2dd47846c2fe027737a5df5e711df1b/687474703a2f2f6936342e74696e797069632e636f6d2f6132727733642e706e67>`_
    and `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/pdf/1409.0473v6.pdf>`_ .
    """
    def __init__(
        self,
        net_encode_in = None,
        net_decode_in = None,
        cell_fn = tf.nn.rnn_cell.LSTMCell,
        cell_init_args = {'state_is_tuple':True},
        n_hidden = 256,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        in_sequence_length = None,
        out_sequence_length = None,
        initial_state = None,
        dropout = None,
        n_layer = 1,
        # return_last = False,
        return_seq_2d = False,
        name = 'attention_seq2seq',
    ):
        Layer.__init__(self, name=name)
        # self.inputs = layer.outputs
        print("  tensorlayer:Instantiate PeekySeq2seq %s: n_hidden:%d, cell_fn:%s, dropout:%s, n_layer:%d" %
              (self.name, n_hidden, cell_fn.__name__, dropout, n_layer))


## Shape layer
class FlattenLayer(Layer):
    """
    The :class:`FlattenLayer` class is layer which reshape high-dimension
    input to a vector. Then we can apply DenseLayer, RNNLayer, ConcatLayer and
    etc on the top of it.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                    act = tf.nn.relu,
    ...                    shape = [5, 5, 32, 64],
    ...                    strides=[1, 1, 1, 1],
    ...                    padding='SAME',
    ...                    name ='cnn_layer')
    >>> network = tl.layers.Pool2dLayer(network,
    ...                    ksize=[1, 2, 2, 1],
    ...                    strides=[1, 2, 2, 1],
    ...                    padding='SAME',
    ...                    pool = tf.nn.max_pool,
    ...                    name ='pool_layer',)
    >>> network = tl.layers.FlattenLayer(network, name='flatten_layer')
    """
    def __init__(
        self,
        layer = None,
        name ='flatten_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = flatten_reshape(self.inputs, name=name)
        self.n_units = int(self.outputs.get_shape()[-1])
        print("  tensorlayer:Instantiate FlattenLayer %s: %d" % (self.name, self.n_units))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


class ReshapeLayer(Layer):
    """
    The :class:`ReshapeLayer` class is layer which reshape the tensor.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    shape : a list
        The output shape.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - The core of this layer is ``tf.reshape``.
    - Use TensorFlow only :
    >>> x = tf.placeholder(tf.float32, shape=[None, 3])
    >>> y = tf.reshape(x, shape=[-1, 3, 3])
    >>> sess = tf.InteractiveSession()
    >>> print(sess.run(y, feed_dict={x:[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]}))
    ... [[[ 1.  1.  1.]
    ... [ 2.  2.  2.]
    ... [ 3.  3.  3.]]
    ... [[ 4.  4.  4.]
    ... [ 5.  5.  5.]
    ... [ 6.  6.  6.]]]
    """
    def __init__(
        self,
        layer = None,
        shape = [],
        name ='reshape_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = tf.reshape(self.inputs, shape=shape, name=name)
        print("  tensorlayer:Instantiate ReshapeLayer %s: %s" % (self.name, self.outputs.get_shape()))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )



class LambdaLayer(Layer):
    """
    The :class:`LambdaLayer` class is a layer which is able to use the provided function.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    fn : a function
        The function that applies to the outputs of previous layer.
    fn_args : a dictionary
        The arguments for the function (option).
    name : a string or None
        An optional name to attach to this layer.

    Examples
    ---------
    >>> x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = LambdaLayer(network, lambda x: 2*x, name='lambda_layer')
    >>> y = network.outputs
    >>> sess = tf.InteractiveSession()
    >>> out = sess.run(y, feed_dict={x : [[1],[2]]})
    ... [[2],[4]]
    """
    def __init__(
        self,
        layer = None,
        fn = None,
        fn_args = {},
        name = 'lambda_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate LambdaLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )

## Merge layer

class ConcatLayer(Layer):
    """
    The :class:`ConcatLayer` class is layer which concat (merge) two or more
    :class:`DenseLayer` to a single class:`DenseLayer`.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    concat_dim : int
        Dimension along which to concatenate.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> sess = tf.InteractiveSession()
    >>> x = tf.placeholder(tf.float32, shape=[None, 784])
    >>> inputs = tl.layers.InputLayer(x, name='input_layer')
    >>> net1 = tl.layers.DenseLayer(inputs, n_units=800, act = tf.nn.relu, name='relu1_1')
    >>> net2 = tl.layers.DenseLayer(inputs, n_units=300, act = tf.nn.relu, name='relu2_1')
    >>> network = tl.layers.ConcatLayer(layer = [net1, net2], name ='concat_layer')
    ...     tensorlayer:Instantiate InputLayer input_layer (?, 784)
    ...     tensorlayer:Instantiate DenseLayer relu1_1: 800, <function relu at 0x1108e41e0>
    ...     tensorlayer:Instantiate DenseLayer relu2_1: 300, <function relu at 0x1108e41e0>
    ...     tensorlayer:Instantiate ConcatLayer concat_layer, 1100
    ...
    >>> tl.layers.initialize_global_variables(sess)
    >>> network.print_params()
    ...     param 0: (784, 800) (mean: 0.000021, median: -0.000020 std: 0.035525)
    ...     param 1: (800,) (mean: 0.000000, median: 0.000000 std: 0.000000)
    ...     param 2: (784, 300) (mean: 0.000000, median: -0.000048 std: 0.042947)
    ...     param 3: (300,) (mean: 0.000000, median: 0.000000 std: 0.000000)
    ...     num of params: 863500
    >>> network.print_layers()
    ...     layer 0: Tensor("Relu:0", shape=(?, 800), dtype=float32)
    ...     layer 1: Tensor("Relu_1:0", shape=(?, 300), dtype=float32)
    ...
    """
    def __init__(
        self,
        layer = [],
        concat_dim = 1,
        name ='concat_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)
        self.outputs = tf.concat(concat_dim, self.inputs, name=name) # 1.2
        self.n_units = int(self.outputs.get_shape()[-1])
        print("  tensorlayer:Instantiate ConcatLayer %s, %d" % (self.name, self.n_units))

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        #self.all_drop = list_remove_repeat(self.all_drop) # it is a dict


class ElementwiseLayer(Layer):
    """
    The :class:`ElementwiseLayer` class combines multiple :class:`Layer` which have the same output shapes by a given elemwise-wise operation.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    combine_fn : a TensorFlow elemwise-merge function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.mul`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`_ .
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - AND Logic
    >>> net_0 = tl.layers.DenseLayer(net_0, n_units=500,
    ...                        act = tf.nn.relu, name='net_0')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=500,
    ...                        act = tf.nn.relu, name='net_1')
    >>> net_com = tl.layers.ElementwiseLayer(layer = [net_0, net_1],
    ...                         combine_fn = tf.minimum,
    ...                         name = 'combine_layer')
    """
    def __init__(
        self,
        layer = [],
        combine_fn = tf.minimum,
        name ='elementwise_layer',
    ):
        Layer.__init__(self, name=name)

        print("  tensorlayer:Instantiate ElementwiseLayer %s:  %s, %s" % (self.name, layer[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layer[0].outputs
        # print(self.outputs._shape, type(self.outputs._shape))
        for l in layer[1:]:
            assert str(self.outputs.get_shape()) == str(l.outputs.get_shape()), "Hint: the input shapes should be the same. %s != %s" %  (self.outputs.get_shape() , str(l.outputs.get_shape()))
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        # self.all_drop = list_remove_repeat(self.all_drop)


# Extend
class ExpandDimsLayer(Layer):
    """
    The :class:`ExpandDimsLayer` class inserts a dimension of 1 into a tensor's shape,
    see `tf.expand_dims() <https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#expand_dims>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    axis : int, 0-D (scalar).
        Specifies the dimension index at which to expand the shape of input.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        axis = None,
        name = 'expand_dims',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate ExpandDimsLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            try:    # TF12
                self.outputs = tf.expand_dims(self.inputs, axis=axis)
            except: # TF11
                self.outputs = tf.expand_dims(self.inputs, dim=axis)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( variables )


class TileLayer(Layer):
    """
    The :class:`TileLayer` class constructs a tensor by tiling a given tensor,
    see `tf.tile() <https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#tile>`_ .

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    multiples: a list of int
        Must be one of the following types: int32, int64. 1-D. Length must be the same as the number of dimensions in input
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = None,
        multiples = None,
        name = 'tile',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate TileLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            self.outputs = tf.tile(self.inputs, multiples=multiples)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( variables )




## TF-Slim layer
class SlimNetsLayer(Layer):
    """
    The :class:`SlimNetsLayer` class can be used to merge all TF-Slim nets into
    TensorLayer. Model can be found in `slim-model <https://github.com/tensorflow/models/tree/master/slim#pre-trained-models>`_ , more about slim
    see `slim-git <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`_ .

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    slim_layer : a slim network function
        The network you want to stack onto, end with ``return net, end_points``.
    slim_args : dictionary
        The arguments for the slim model.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - see Inception V3 example on `Github <https://github.com/zsdonghao/tensorlayer>`_

    Notes
    -----
    The due to TF-Slim stores the layers as dictionary, the ``all_layers`` in this
    network is not in order ! Fortunately, the ``all_params`` are in order.
    """
    def __init__(
        self,
        layer = None,
        slim_layer = None,
        slim_args = {},
        name ='InceptionV3',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate SlimNetsLayer %s: %s" % (self.name, slim_layer.__name__))

        # with tf.variable_scope(name) as vs:
        #     net, end_points = slim_layer(self.inputs, **slim_args)
        #     slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        net, end_points = slim_layer(self.inputs, **slim_args)

        slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=name)
        if slim_variables == []:
            print("No variables found under %s : the name of SlimNetsLayer should be matched with the begining of the ckpt file, see tutorial_inceptionV3_tfslim.py for more details" % name)


        self.outputs = net

        slim_layers = []
        for v in end_points.values():
            # tf.contrib.layers.summaries.summarize_activation(v)
            slim_layers.append(v)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( slim_layers )
        self.all_params.extend( slim_variables )

## Special activation
class PReluLayer(Layer):
    """
    The :class:`PReluLayer` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    channel_shared : `bool`. Single weight is shared by all channels
    a_init : alpha initializer, default zero constant.
        The initializer for initializing the alphas.
    a_init_args : dictionary
        The arguments for the weights initializer.
    name : A name for this activation op (optional).

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """
    def __init__(
        self,
        layer = None,
        channel_shared = False,
        a_init = tf.constant_initializer(value=0.0),
        a_init_args = {},
        # restore = True,
        name="prelu_layer"
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate PReluLayer %s: channel_shared:%s" % (self.name, channel_shared))
        if channel_shared:
            w_shape = (1,)
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name) as vs:
            alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=a_init, **a_init_args )
            self.outputs = tf.nn.relu(self.inputs) + tf.mul(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [alphas] )


## Flow control layer
class MultiplexerLayer(Layer):
    """
    The :class:`MultiplexerLayer` selects one of several input and forwards the selected input into the output,
    see `tutorial_mnist_multiplexer.py`.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.


    Variables
    -----------------------
    sel : a placeholder
        Input an int [0, inf], which input is the output

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    >>> y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
    >>> # define the network
    >>> net_in = tl.layers.InputLayer(x, name='input_layer')
    >>> net_in = tl.layers.DropoutLayer(net_in, keep=0.8, name='drop1')
    >>> # net 0
    >>> net_0 = tl.layers.DenseLayer(net_in, n_units=800,
    ...                                act = tf.nn.relu, name='net0/relu1')
    >>> net_0 = tl.layers.DropoutLayer(net_0, keep=0.5, name='net0/drop2')
    >>> net_0 = tl.layers.DenseLayer(net_0, n_units=800,
    ...                                act = tf.nn.relu, name='net0/relu2')
    >>> # net 1
    >>> net_1 = tl.layers.DenseLayer(net_in, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu1')
    >>> net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop2')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu2')
    >>> net_1 = tl.layers.DropoutLayer(net_1, keep=0.8, name='net1/drop3')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=800,
    ...                                act = tf.nn.relu, name='net1/relu3')
    >>> # multiplexer
    >>> net_mux = tl.layers.MultiplexerLayer(layer = [net_0, net_1], name='mux_layer')
    >>> network = tl.layers.ReshapeLayer(net_mux, shape=[-1, 800], name='reshape_layer') #
    >>> network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    >>> # output layer
    >>> network = tl.layers.DenseLayer(network, n_units=10,
    ...                                act = tf.identity, name='output_layer')

    References
    ------------
    - See ``tf.pack()`` and ``tf.gather()`` at `TensorFlow - Slicing and Joining <https://www.tensorflow.org/versions/master/api_docs/python/array_ops.html#slicing-and-joining>`_
    """
    def __init__(self,
               layer = [],
               name='mux_layer'):
        Layer.__init__(self, name=name)
        self.n_inputs = len(layer)

        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)
        all_inputs = tf.pack(self.inputs, name=name) # pack means concat a list of tensor in a new dim  # 1.2

        print("  tensorlayer:Instantiate MultiplexerLayer %s: n_inputs: %d" % (self.name, self.n_inputs))

        self.sel = tf.placeholder(tf.int32)
        self.outputs = tf.gather(all_inputs, self.sel, name=name) # [sel, :, : ...] # 1.2

        # print(self.outputs, vars(self.outputs))
        #         # tf.reshape(self.outputs, shape=)
        # exit()
        # the same with ConcatLayer
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        # self.all_drop = list_remove_repeat(self.all_drop)
## We can Duplicate the network instead of DemultiplexerLayer
# class DemultiplexerLayer(Layer):
#     """
#     The :class:`DemultiplexerLayer` takes a single input and select one of many output lines, which is connected to the input.
#
#     Parameters
#     ----------
#     layer : a list of :class:`Layer` instances
#         The `Layer` class feeding into this layer.
#     n_outputs : a int
#         The number of output
#     name : a string or None
#         An optional name to attach to this layer.
#
#     Field (Class Variables)
#     -----------------------
#     sel : a placeholder
#         Input int [0, inf], the
#     outputs : a list of Tensor
#         A list of outputs
#
#     Examples
#     --------
#     >>>
#     """
#     def __init__(self,
#            layer = None,
#            name='demux_layer'):
#         Layer.__init__(self, name=name)
#         self.outputs = []

## Wrapper
class EmbeddingAttentionSeq2seqWrapper(Layer):
  """Sequence-to-sequence model with attention and for multiple buckets.

    This example implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper:
    - `Grammar as a Foreign Language <http://arxiv.org/abs/1412.7449>`_
    please look there for details,
    or into the seq2seq library for complete model implementation.
    This example also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
    - `Neural Machine Translation by Jointly Learning to Align and Translate <http://arxiv.org/abs/1409.0473>`_
    The sampled softmax is described in Section 3 of the following paper.
    - `On Using Very Large Target Vocabulary for Neural Machine Translation <http://arxiv.org/abs/1412.2007>`_

    Parameters
    ----------
    source_vocab_size : size of the source vocabulary.
    target_vocab_size : size of the target vocabulary.
    buckets : a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
    size : number of units in each layer of the model.
    num_layers : number of layers in the model.
    max_gradient_norm : gradients will be clipped to maximally this norm.
    batch_size : the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
    learning_rate : learning rate to start with.
    learning_rate_decay_factor : decay learning rate by this much when needed.
    use_lstm : if true, we use LSTM cells instead of GRU cells.
    num_samples : number of samples for sampled softmax.
    forward_only : if set, we do not construct the backward pass in the model.
    name : a string or None
        An optional name to attach to this layer.
  """
  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               name='wrapper'):
    Layer.__init__(self)#, name=name)

    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False, name='global_step')

    # =========== Fake output Layer for compute cost ======
    # If we use sampled softmax, we need an output projection.
    with tf.variable_scope(name) as vs:
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
          w = tf.get_variable("proj_w", [size, self.target_vocab_size])
          w_t = tf.transpose(w)
          b = tf.get_variable("proj_b", [self.target_vocab_size])
          output_projection = (w, b)

          def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                    self.target_vocab_size)
          softmax_loss_function = sampled_loss

        # ============ Seq Encode Layer =============
        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size)
        if use_lstm:
          single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
        cell = single_cell
        if num_layers > 1:
          cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # ============== Seq Decode Layer ============
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return tf.nn.seq2seq.embedding_attention_seq2seq(
              encoder_inputs, decoder_inputs, cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode)

        #=============================================================
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]
        self.targets = targets  # DH add for debug


        # Training outputs and losses.
        if forward_only:
          self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
              softmax_loss_function=softmax_loss_function)
          # If we use output projection, we need to project outputs for decoding.
          if output_projection is not None:
            for b in xrange(len(buckets)):
              self.outputs[b] = [
                  tf.matmul(output, output_projection[0]) + output_projection[1]
                  for output in self.outputs[b]
              ]
        else:
          self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
              self.encoder_inputs, self.decoder_inputs, targets,
              self.target_weights, buckets,
              lambda x, y: seq2seq_f(x, y, False),
              softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
          self.gradient_norms = []
          self.updates = []
          opt = tf.train.GradientDescentOptimizer(self.learning_rate)
          for b in xrange(len(buckets)):
            gradients = tf.gradients(self.losses[b], params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step))

        # if save into npz
        self.all_params = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

    # if save into ckpt
    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Parameters
    ----------
    session : tensorflow session to use.
    encoder_inputs : list of numpy int vectors to feed as encoder inputs.
    decoder_inputs : list of numpy int vectors to feed as decoder inputs.
    target_weights : list of numpy float vectors to feed as target weights.
    bucket_id : which bucket of the model to use.
    forward_only : whether to do the backward step or only forward.

    Returns
    --------
    A triple consisting of gradient norm (or None if we did not do backward),
    average perplexity, and the outputs.

    Raises
    --------
    ValueError : if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
    # print('in model.step()')
    # print('a',bucket_id, encoder_size, decoder_size)

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]
    # print(self.encoder_inputs[l].name)
    # print(self.decoder_inputs[l].name)
    # print(self.target_weights[l].name)

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
    # print('last_target', last_target)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id, PAD_ID=0, GO_ID=1, EOS_ID=2, UNK_ID=3):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Parameters
    ----------
    data : a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
    bucket_id : integer, which bucket to get the batch for.
    PAD_ID : int
        Index of Padding in vocabulary
    GO_ID : int
        Index of GO in vocabulary
    EOS_ID : int
        Index of End of sentence in vocabulary
    UNK_ID : int
        Index of Unknown word in vocabulary

    Returns
    -------
    The triple (encoder_inputs, decoder_inputs, target_weights) for
    the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights



## Developing or Untested
class MaxoutLayer(Layer):
    """
    Waiting for contribution

    Single DenseLayer with Max-out behaviour, work well with Dropout.

    References
    -----------
    `Goodfellow (2013) Maxout Networks <http://arxiv.org/abs/1302.4389>`_
    """
    def __init__(
        self,
        layer = None,
        n_units = 100,
        name ='maxout_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate MaxoutLayer %s: %d" % (self.name, self.n_units))
        print("    Waiting for contribution")
        with tf.variable_scope(name) as vs:
            pass
            # W = tf.Variable(init.xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
            # b = tf.Variable(tf.zeros([n_units]), name='b')

        # self.outputs = act(tf.matmul(self.inputs, W) + b)
        # https://www.tensorflow.org/versions/r0.9/api_docs/python/array_ops.html#pack
        # http://stackoverflow.com/questions/34362193/how-to-explicitly-broadcast-a-tensor-to-match-anothers-shape-in-tensorflow
        # tf.concat tf.pack  tf.tile

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )
























#
