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

## Dynamically creat variables for keep prob
# set_keep = locals()
set_keep = globals()
set_keep['_layers_name_list'] =[]
set_keep['name_reuse'] = False

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

    Examples
    ------------
    - see ``tutorial_ptb_lstm.py`` for example.
    """
    set_keep['name_reuse'] = enable

def initialize_rnn_state(state):
    """Return the initialized RNN state.
    The input is LSTMStateTuple or State of RNNCells.
    """
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        # when state_is_tuple=True for LSTM
        # print(state)
        # print(state.c)
        # print(state.h)
        # print(state.c.eval())
        # print(state.h.eval())
        # exit()
        c = state.c.eval()
        h = state.h.eval()
        return (c, h)
        # # print(state)
        # # print(state[0])
        # new_state = state
        # new_state[0].assign(state[0].eval())
        # new_state[1].assign(state[1].eval())
        # # state[0] = state[0].eval()
        # # state[1] = state[1].eval()
        # # state.c = state.c.eval()
        # # state.h = state.h.eval()
        # return new_state
    else:
        # when state_is_tuple=False for LSTM
        # or other RNNs
        new_state = state.eval()
        return new_state


def print_all_variables(train_only=False):
    """Print all trainable and non-trainable variables
    without initialize_all_variables()

    Parameters
    ----------
    train_only : boolen
        If True, only print the trainable variables, otherwise, print all variables.
    """
    tvar = tf.trainable_variables() if train_only else tf.all_variables()
    for idx, v in enumerate(tvar):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

# def print_all_variables():
#     """Print all trainable and non-trainable variables
#     without initialize_all_variables()"""
#     for idx, v in enumerate(tf.all_variables()):
#         # print("  var %d: %s   %s" % (idx, v.get_shape(), v.name))
#         print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

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
        # if name in globals():
        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name'.\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)


    def print_params(self, details=True):
        ''' Print all info of parameters in the network'''
        # try:
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    # print("  param %d: %s (mean: %f, median: %f, std: %f)   %s" % (i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                    print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                except:
                    raise Exception("Hint: print params details after sess.run(tf.initialize_all_variables()) or use network.print_params(False).")
            else:
                print("  param {:3}: {:15}    {}".format(i, str(p.get_shape()), p.name))
        print("  num of params: %d" % self.count_params())
        # except:
        #     raise Exception("Hint: print params after sess.run(tf.initialize_all_variables()) or use tl.layers.print_all_variables()")


    def print_layers(self):
        ''' Print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            # print(vars(p))
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

    # def print_params(self):
    #     ''' Print all info of parameters in the network after initialize_all_variables()'''
    #     try:
    #         for i, p in enumerate(self.all_params):
    #             print("  param %d: %s (mean: %f, median: %f, std: %f)   %s" % (i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
    #         print("  num of params: %d" % self.count_params())
    #     except:
    #         raise Exception("Hint: print params after sess.run(tf.initialize_all_variables()) or use tl.layers.print_all_variables()")
    #
    #
    # def print_layers(self):
    #     ''' Print all info of layers in the network '''
    #     for i, p in enumerate(self.all_layers):
    #         # print(vars(p))
    #         print("  layer %d: %s" % (i, str(p)))
    #
    # def count_params(self):
    #     ''' Return the number of parameters in the network '''
    #     n_params = 0
    #     for i, p in enumerate(self.all_params):
    #         n = 1
    #         for s in p.eval().shape:
    #         # for s in p.get_shape():
    #             # s = int(s)
    #             if s:
    #                 n = n * s
    #         n_params = n_params + n
    #     return n_params

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
        # super(InputLayer, self).__init__()     # initialize all super classes
        # if n_features:
        #     self.n_units = n_features
        # else:
        #     self.n_units = int(inputs._shape[1])
        print("  tensorlayer:Instantiate InputLayer  %s: %s" % (self.name, inputs._shape))

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
    ...         nce_loss_args = {},
    ...         E_init = tf.random_uniform,
    ...         E_init_args = {'minval':-1.0, 'maxval':1.0},
    ...         nce_W_init = tf.truncated_normal,
    ...         nce_W_init_args = {'stddev': float(1.0/np.sqrt(embedding_size))},
    ...         nce_b_init = tf.zeros,
    ...         nce_b_init_args = {},
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
        self.n_units = embedding_size
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

    This class can not be used to train a word embedding matrix, so you should
    assign a trained matrix into it. To train a word embedding matrix, you can used
    class:`Word2vecEmbeddingInputlayer`.

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
    >>> sess.run(tf.initialize_all_variables())
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
        self.n_units = embedding_size
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
        act = tf.nn.relu,
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

        n_in = int(self.inputs._shape[-1])
        self.n_units = n_units
        print("  tensorlayer:Instantiate DenseLayer  %s: %d, %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
            if b_init:
                b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))
        # self.outputs = act(tf.matmul(self.inputs, W) + b)

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
        # shallow cope allows the weights in network can be changed at the same
        # time, when ReconLayer updates the weights of encoder.
        #
        # e.g. the encoder points to same physical memory address
        # network = InputLayer(x, name='input_layer')
        # network = DenseLayer(network, n_units=200, act = tf.nn.sigmoid, name='sigmoid')
        # recon_layer = ReconLayer(network, n_units=784, act = tf.nn.sigmoid, name='recon_layer')
        # print(network.all_params)
        #   [<tensorflow.python.ops.variables.Variable object at 0x10d616f98>,
        #   <tensorflow.python.ops.variables.Variable object at 0x10d8f6080>]
        # print(len(network.all_params))
        #   2
        # print(recon_layer.all_params)
        #   [<tensorflow.python.ops.variables.Variable object at 0x10d616f98>,
        #   <tensorflow.python.ops.variables.Variable object at 0x10d8f6080>,
        #   <tensorflow.python.ops.variables.Variable object at 0x10d8f6550>,
        #   <tensorflow.python.ops.variables.Variable object at 0x10d8f6198>]
        # print(len(recon_layer.all_params))
        #   4

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
                        raise Exception("You should change visualize.W(), if you want to save the feature images for different dataset")


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
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    >>> network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        name = 'dropout_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate DropoutLayer %s: keep: %f" % (self.name, keep))

        # The name of placeholder for keep_prob is the same with the name
        # of the Layer.
        set_keep[name] = tf.placeholder(tf.float32)
        self.outputs = tf.nn.dropout(self.inputs, set_keep[name], name=name) # 1.2

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
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
        act = tf.nn.relu,
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
        n_in = int(self.inputs._shape[-1])
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


## Convolutional layer
class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see ``tf.nn.conv2d``.

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

    References
    ----------
    - `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`_
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
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
    The :class:`DeConv2dLayer` class is deconvolutional 2D layer, see ``tf.nn.conv2d_transpose``.

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

    Examples
    ---------
    - A part of the generator in DCGAN example
    >>> inputs = tf.placeholder(tf.float32, [64, 100], name='z_noise')
    >>> net_in = tl.layers.InputLayer(inputs, name='g/in')
    >>> net_h0 = tl.layers.DenseLayer(net_in, n_units = 8192,
    ...                            W_init = tf.random_normal_initializer(stddev=0.02),
    ...                            act = tf.identity, name='g/h0/lin')
    >>> print(net_h0.outputs)
    ... (64, 8192)
    >>> net_h0 = tl.layers.ReshapeLayer(net_h0, shape = [-1, 4, 4, 512], name='g/h0/reshape')
    >>> net_h0 = tl.layers.BatchNormLayer(net_h0, is_train=is_train, name='g/h0/batch_norm')
    >>> net_h0.outputs = tf.nn.relu(net_h0.outputs, name='g/h0/relu')
    >>> print(net_h0.outputs)
    ... (64, 4, 4, 512)
    >>> net_h1 = tl.layers.DeConv2dLayer(net_h0,
    ...                            shape = [5, 5, 256, 512],
    ...                            output_shape = [64, 8, 8, 256],
    ...                            strides=[1, 2, 2, 1],
    ...                            act=tf.identity, name='g/h1/decon2d')
    >>> net_h1 = tl.layers.BatchNormLayer(net_h1, is_train=is_train, name='g/h1/batch_norm')
    >>> net_h1.outputs = tf.nn.relu(net_h1.outputs, name='g/h1/relu')
    >>> print(net_h1.outputs)
    ... (64, 8, 8, 256)

    References
    ----------
    - `tf.nn.conv2d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d_transpose>`_
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
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
                            (self.name, str(shape), str(output_shape), str(strides), padding, act))
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
    The :class:`Conv3dLayer` class is a 3D CNN layer, see ``tf.nn.conv3d``.

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

    References
    ----------
    - `tf.nn.conv3d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d>`_
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
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
    """
    The :class:`DeConv3dLayer` class is deconvolutional 3D layer, see ``tf.nn.conv3d_transpose``.

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

    References
    ----------
    - `tf.nn.conv3d_transpose <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv3d_transpose>`_
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
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
        print("  tensorlayer:Instantiate DeConv2dLayer %s: %s, %s, %s, %s, %s" %
                            (self.name, str(shape), str(output_shape), str(strides), padding, act))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_deconv3d', shape=shape, initializer=W_init, **W_init_args )
            b = tf.get_variable(name='b_deconv3d', shape=(shape[-2]), initializer=b_init, **b_init_args )

        self.outputs = act( tf.nn.conv3d_transpose(self.inputs, W, output_shape=output_shape, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

## Normalization layer
class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float
        A decay factor for ExponentialMovingAverage.
    epsilon : float
        A small float number to avoid dividing by 0.
    is_train : boolen
        Whether train or inference.
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `tf.nn.batch_normalization <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.batch_normalization.md>`_
    - `stackoverflow <http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow>`_
    - `tensorflow.contrib <https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100>`_
    """
    def __init__(
        self,
        layer = None,
        decay = 0.999,
        epsilon = 0.001,
        is_train = None,
        name ='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate BatchNormLayer %s: decay: %f, epsilon: %f, is_train: %s" %
                            (self.name, decay, epsilon, is_train))
        if is_train == None:
            raise Exception("is_train must be True or False")

        # (name, input_var, decay, epsilon, is_train)
        inputs_shape = self.inputs.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]

        with tf.variable_scope(name) as vs:
            beta = tf.get_variable(name='beta', shape=params_shape,
                                 initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable(name='gamma', shape=params_shape,
                                  initializer=tf.constant_initializer(1.0))
            batch_mean, batch_var = tf.nn.moments(self.inputs,
                                                axis,
                                                name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
              ema_apply_op = ema.apply([batch_mean, batch_var])
              with tf.control_dependencies([ema_apply_op]):
                  return tf.identity(batch_mean), tf.identity(batch_var)

            if is_train:
                is_train = tf.cast(tf.ones(1), tf.bool)
            else:
                is_train = tf.cast(tf.zeros(1), tf.bool)

            is_train = tf.reshape(is_train, [])

            # print(is_train)
            # exit()

            mean, var = tf.cond(
              is_train,
              mean_var_with_update,
              lambda: (ema.average(batch_mean), ema.average(batch_var))
            )
            normed = tf.nn.batch_normalization(
              x=self.inputs,
              mean=mean,
              variance=var,
              offset=beta,
              scale=gamma,
              variance_epsilon=epsilon,
              name='tf_bn'
            )
        self.outputs = normed

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [beta, gamma] )


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
        tf.nn.max_pool , tf.nn.avg_pool ...
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - see Conv2dLayer

    References
    ----------
    - `TensorFlow Pooling <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`_
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
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
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
    return_last : boolen
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
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
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    References
    ----------
    - `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
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
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")


        # is_reshape : boolen (deprecate)
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
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

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

# Dynamic RNN

def advanced_indexing_op(input, index):
    """ Advanced Indexing for Sequences. see TFlearn."""
    batch_size = tf.shape(input)[0]
    max_length = int(input.get_shape()[1])
    dim_size = int(input.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(input, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant

def retrieve_seq_length_op(data):
    """ An op to compute the length of a sequence. 0 are masked. see TFlearn."""
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
    return length

class DynamicRNNLayer(Layer):
    """
    The :class:`DynamicRNNLayer` class is a Dynamic RNN layer, see ``tf.nn.dynamic_rnn``.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN cell as follow.
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
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
        The sequence length of each row of input data. If None, automatically calculate the sequence length for the data.
    initial_state : None or RNN State
        If None, initial_state is zero_state.
    dropout : `tuple` of `float`: (input_keep_prob, output_keep_prob).
        The input and output keep probability.
    n_layer : a int, default is 1.
        The number of RNN layers.
    return_last : boolen
        - If True, return the last output, "Sequence input and single output"
        - If False, return all outputs, "Synced sequence input and output"
        - In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
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

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    Examples
    --------
    >>> input_feed = tf.placeholder(dtype=tf.int64,
    ...                              shape=[None],  # word id
    ...                              name="input_feed")
    >>> input_seqs = tf.expand_dims(input_feed, 1)
    >>> network = tl.layers.EmbeddingInputlayer(
    ...             inputs = input_seqs,
    ...             vocabulary_size = vocab_size,
    ...             embedding_size = embedding_size,
    ...             name = 'seq_embedding')
    >>> network = tl.layers.DynamicRNNLayer(network,
    ...             cell_fn = tf.nn.rnn_cell.BasicLSTMCell,
    ...             n_hidden = embedding_size,
    ...             dropout = 0.7,
    ...             return_seq_2d = True,     # stack denselayer or compute cost after it
    ...             name = 'dynamic_rnn',)
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
        n_hidden = 64,
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

        print("  tensorlayer:Instantiate DynamicRNNLayer %s: n_hidden:%d, in_dim:%d %s, cell_fn:%s " % (self.name, n_hidden,
             self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))

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
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * n_layer, state_is_tuple=True)

        # Initialize initial_state
        if initial_state is None:
            self.initial_state = self.cell.zero_state(batch_size, dtype=tf.float32)#dtype="float")
        else:
            self.initial_state = initial_state

        # Computes sequence_length
        if sequence_length is None:
            sequence_length = retrieve_seq_length_op(
                        self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))
        # print('sequence_length',sequence_length)

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
            # result = tf.contrib.learn.run_n(
            #     {"outputs": outputs, "last_states": last_states}, n=1, feed_dict=None)
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))
        # exit()
        # Manage the outputs
        if return_last:
            # [batch_size, n_hidden]
            # outputs = tf.transpose(tf.pack(result[0]["outputs"]), [1, 0, 2])
            outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
            self.outputs = advanced_indexing_op(outputs, sequence_length)
        else:
            # [batch_size, n_step(max), n_hidden]
            # self.outputs = result[0]["outputs"]
            self.outputs = outputs
            if return_seq_2d:
                # PTB tutorial:
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, n_hidden])
            # else:
                # <akara>:
                # 3D Tensor [batch_size, n_steps, n_hidden]
                # self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, n_steps, n_hidden])


        # Final state
        # self.final_state = result[0]["last_states"]
        self.final_state = last_states
        # print(self.final_state)
        # exit()

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
        - see `RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_\n
        - class ``tf.nn.rnn_cell.BasicRNNCell``
        - class ``tf.nn.rnn_cell.BasicLSTMCell``
        - class ``tf.nn.rnn_cell.GRUCell``
        - class ``tf.nn.rnn_cell.LSTMCell``
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    n_hidden : a int
        The number of hidden units in the layer.
    n_steps : a int
        The sequence length.
    return_last : boolen
        If True, return the last output, "Sequence input and single output"\n
        If False, return all outputs, "Synced sequence input and output"\n
        In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    return_seq_2d : boolen
        When return_last = False\n
            if True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
            if False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    name : a string or None
        An optional name to attach to this layer.

    Variables
    -----------------------
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
        cell_init_args = {'state_is_tuple' : True},
        n_hidden = 64,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        # n_steps = 5,
        return_last = False,
        # is_reshape = True,
        return_seq_2d = False,
        name = 'birnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate BiDynamicRNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, cell_fn:%s " % (self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        print("     Untested !!!")

        self.cell = cell = cell_fn(num_units=n_hidden, **cell_init_args)
        # self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        # state = self.initial_state

        with tf.variable_scope(name, initializer=initializer) as vs:
            outputs, states  = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell,
                    cell_bw=cell,
                    dtype=tf.float64,
                    sequence_length=X_lengths,
                    inputs=X)

            output_fw, output_bw = outputs
            states_fw, states_bw = states

            result = tf.contrib.learn.run_n(
                {"output_fw": output_fw, "output_bw": output_bw, "states_fw": states_fw, "states_bw": states_bw},
                n=1,
                feed_dict=None)
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = output_fw
        else:
            if return_seq_2d:
                # PTB tutorial:
                # 2D Tensor [n_example, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, output_fw), [-1, n_hidden])
            else:
                # <akara>:
                # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                self.outputs = tf.reshape(tf.concat(1, output_fw), [-1, n_steps, n_hidden])

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )





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
        self.n_units = int(self.outputs._shape[-1])
        print("  tensorlayer:Instantiate FlattenLayer %s: %d" % (self.name, self.n_units))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )

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
    >>> sess.run(tf.initialize_all_variables())
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
        self.n_units = int(self.outputs._shape[-1])
        print("  tensorlayer:Instantiate ConcatLayer %s, %d" % (self.name, self.n_units))

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))


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
        print("  tensorlayer:Instantiate ReshapeLayer %s: %s" % (self.name, self.outputs._shape))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )

## TF-Slim layer
class SlimNetsLayer(Layer):
    """
    The :class:`SlimNetsLayer` class can be used to merge all TF-Slim nets into
    TensorLayer. Model can be found in `slim-model <https://github.com/tensorflow/models/tree/master/slim#Install>`_ , more about slim
    see `slim-git <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim>`_ .

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    slim_layer : a slim network function
        The network you want to stack onto, end with ``return net, end_points``.
    name : a string or None
        An optional name to attach to this layer.

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
        name ='slim_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate SlimNetsLayer %s: %s" % (self.name, slim_layer.__name__))

        with tf.variable_scope(name) as vs:
            net, end_points = slim_layer(self.inputs, **slim_args)
            slim_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

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
    W_init: weights initializer, default zero constant.
        The initializer for initializing the alphas.
    restore : `bool`. Restore or not alphas
    name : A name for this activation op (optional).

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """
    def __init__(
        self,
        layer = None,
        channel_shared = False,
        W_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        restore = True,
        name="prelu_layer"
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate PReluLayer %s: %s" % (self.name, channel_shared))
        print('     [Warning] prelu: untested !!!')
        if channel_shared:
            w_shape = (1,)
        else:
            w_shape = int(self.inputs._shape[-1:])

        with tf.name_scope(name) as scope:
            # W_init = initializations.get(weights_init)()
            alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=W_init, **W_init_args )
            self.outputs = tf.nn.relu(self.inputs) + tf.mul(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend( self.outputs )
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
        self.all_params = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

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

# noise
class GaussianNoiseLayer(Layer):
    """
    Waiting for contribution
    """
    def __init__(
        self,
        layer = None,
        # keep = 0.5,
        name = 'gaussian_noise_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  tensorlayer:Instantiate GaussianNoiseLayer %s: keep: %f" % (self.name, keep))
        print("    Waiting for contribution")
        with tf.variable_scope(name) as vs:
            pass






















#
