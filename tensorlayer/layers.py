#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import time
from . import init
from . import visualize
from . import utils
from . import files
from . import cost
from . import iterate
import numpy as np

# __all__ = [
#     "Layer",
#     "DenseLayer",
# ]

## Dynamically creat variable for keep prob
# set_keep = locals()
set_keep = globals()
set_keep['_layers_name_list'] =[]

## Variable Operation
def flatten_reshape(variable):
    """Reshapes high-dimension input to a vector.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    variable : a tensorflow variable

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
    return tf.reshape(variable, shape=[-1, dim])

# Basic layer
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
        if name in set_keep['_layers_name_list']:
            raise Exception("Layer '%s' already exists, please choice other 'name'.\
            \nHint : Use different name for different 'Layer'" % name)
        else:
            self.name = name
            set_keep['_layers_name_list'].append(name)

    # @staticmethod


    # @instancemethod
    def print_params(self):
        ''' print all info of parameters in the network '''
        for i, p in enumerate(self.all_params):
            print("  param %d: %s (mean: %f, median: %f std: %f)" % (i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std()))
        print("  num of params: %d" % self.count_params())

    # @instancemethod
    def print_layers(self):
        ''' print all info of layers in the network '''
        for i, p in enumerate(self.all_layers):
            # print(vars(p))
            print("  layer %d: %s" % (i, str(p)))


    def count_params(self):
        ''' return the number of parameters in the network '''
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            for s in p.eval().shape:
            # for s in p.get_shape():
                # s = int(s)
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

# Input layer
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
        # super(InputLayer, self).__init__()            # initialize all super classes
        if n_features:
            self.n_units = n_features
        else:
            self.n_units = int(inputs._shape[1])
        print("  tensorlayer:Instantiate InputLayer %s %s" % (self.name, inputs._shape))

        self.outputs = inputs

        self.all_layers = []
        self.all_params = []
        self.all_drop = {}

# Word Embedding Input layer
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
        The initializer for initializing the nce decoder bias vector.
    nce_b_init_args : a dictionary
        The arguments for initializing the nce decoder bias vector.
    name : a string or None
        An optional name to attach to this layer.

    Field (Class Variables)
    -----------------------
    nce_cost : a tensor
        The NCE loss.
    outputs : a tensor
        The outputs of embedding layer.
    normalized_embeddings : tensor
        Normalized embedding matrix

    Examples
    --------
    >>> Without TensorLayer : see tensorflow/examples/tutorials/word2vec/word2vec_basic.py
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

    >>> With TensorLayer : see tutorial_word2vec_basic.py
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
    >>> train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, var_list=train_params)
    >>> normalized_embeddings = emb_net.normalized_embeddings

    References
    ----------
    `tensorflow/examples/tutorials/word2vec/word2vec_basic.py <https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py>`_
    """
    def __init__(
        self,
        inputs = None,
        train_labels = None,
        vocabulary_size = 80000,
        embedding_size = 200,
        num_sampled = 64,
        nce_loss_args = {},
        E_init = tf.random_uniform,
        E_init_args = {'minval':-1.0, 'maxval':1.0},
        nce_W_init = tf.truncated_normal,
        nce_W_init_args = {'stddev':0.03},
        nce_b_init = tf.zeros,
        nce_b_init_args = {},
        name ='word2vec_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs#layer.outputs
        # n_in = layer.n_units
        self.n_units = embedding_size
        print("  tensorlayer:Instantiate Word2vecEmbeddingInputlayer %s (%d, %d)" % (self.name, vocabulary_size, embedding_size))
        # Look up embeddings for inputs.
        # Note: a row of 'embeddings' is the vector representation of a word.
        # for the sake of speed, it is better to slice the embedding matrix
        # instead of transfering a word id to one-hot-format vector and then
        # multiply by the embedding matrix.
        # embed is the outputs of the hidden layer (embedding layer), it is a
        # row vector with 'embedding_size' values.
        embeddings = tf.Variable(
            E_init(shape=[vocabulary_size, embedding_size], **E_init_args))
        embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        # Construct the variables for the NCE loss (i.e. negative sampling)
        nce_weights = tf.Variable(
            nce_W_init(shape=[vocabulary_size, embedding_size], **nce_W_init_args))
        nce_biases = tf.Variable(nce_b_init([vocabulary_size], **nce_b_init_args))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels
        # each time we evaluate the loss.
        self.nce_cost = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                           inputs=embed, labels=train_labels,
                           num_sampled=num_sampled, num_classes=vocabulary_size,
                           **nce_loss_args))
        # num_sampled: An int. The number of classes to randomly sample per batch
        #              Number of negative examples to sample.
        # num_classes: An int. The number of possible classes.
        # num_true = 1: An int. The number of target classes per training example.
        #            DH: if 1, predict one word given one word, like bigram model?  Check!

        self.outputs = embed
        self.normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)


        self.all_layers = [self.outputs]
        self.all_params = [embeddings, nce_weights, nce_biases]
        self.all_drop = {}

        # self.all_layers = list(layer.all_layers)    # list() is pass by value (shallow), without list is pass by reference
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)        # dict() is pass by value (shallow), without dict is pass by reference
        # self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( [embeddings, nce_weights, nce_biases] )

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
    vocabulary_size : int
        The size of vocabulary, number of words.
    embedding_size : int
        The number of embedding dimensions.
    name : a string or None
        An optional name to attach to this layer.

    Field (Class Variables)
    -----------------------
    outputs : a tensor
        The outputs of embedding layer.

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
        name ='embedding_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = inputs
        self.n_units = embedding_size
        print("  tensorlayer:Instantiate EmbeddingInputlayer %s (%d, %d)" % (self.name, vocabulary_size, embedding_size))

        embeddings = tf.Variable(
            tf.random_uniform(shape=[vocabulary_size, embedding_size]))
        embed = tf.nn.embedding_lookup(embeddings, self.inputs)

        self.outputs = embed

        self.all_layers = [self.outputs]
        self.all_params = [embeddings]
        self.all_drop = {}



# Dense layer
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
    b_init : biases initializer
        The initializer for initializing the bias vector.
    W_init_args : dictionary
        The arguments for the weights initializer.
    b_init_args : dictionary
        The arguments for the biases initializer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1',
    ...            W_init=tf.random_normal, W_init_args={'mean':1.0, 'stddev':1.0})

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
        W_init = init.xavier_init,
        b_init = tf.zeros,
        W_init_args = {},
        b_init_args = {},
        name ='dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_units
        print("  tensorlayer:Instantiate DenseLayer %s: %d, %s" % (self.name, self.n_units, act))

        W = tf.Variable(W_init(shape=(n_in, n_units), **W_init_args), name='W')
        b = tf.Variable(b_init(shape=[n_units], **b_init_args), name='b')
        self.outputs = act(tf.matmul(self.inputs, W) + b)

        self.all_layers = list(layer.all_layers)    # list() is pass by value (shallow), without list is pass by reference
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)        # dict() is pass by value (shallow), without dict is pass by reference
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )
        # shallow cope, when ReconLayer updates the weights of encoder, the weights in network can be changed at the same time.
        # e.g. the encoder points to same physical memory address
        # network = InputLayer(x, name='input_layer')
        # network = DenseLayer(network, n_units=200, act = tf.nn.sigmoid, name='sigmoid')
        # recon_layer = ReconLayer(network, n_units=784, act = tf.nn.sigmoid, name='recon_layer')
        # print(network.all_params)             [<tensorflow.python.ops.variables.Variable object at 0x10d616f98>, <tensorflow.python.ops.variables.Variable object at 0x10d8f6080>]
        # print(len(network.all_params))        2
        # print(recon_layer.all_params)         [<tensorflow.python.ops.variables.Variable object at 0x10d616f98>, <tensorflow.python.ops.variables.Variable object at 0x10d8f6080>, <tensorflow.python.ops.variables.Variable object at 0x10d8f6550>, <tensorflow.python.ops.variables.Variable object at 0x10d8f6198>]
        # print(len(recon_layer.all_params))    4

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
    >>> network = tl.layers.DenseLayer(network, n_units=196, act=tf.nn.sigmoid, name='sigmoid1')
    >>> recon_layer1 = tl.layers.ReconLayer(network, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')
    >>> recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')

    Methods
    -------
    pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
        Start to pre-train the parameters of previous DenseLayer.

    Notes
    -----
    The input layer should be `DenseLayer` or a layer has only one axes.
    You may need to modify this part to define your own cost function.
    By default, the cost is implemented as follow:

    For sigmoid layer, the implementation can be `UFLDL <http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial>`_

    For rectifying layer, the implementation can be `Glorot (2011). Deep Sparse Rectifier Neural Networks <http://doi.org/10.1.1.208.6449>`_

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
        print("     tensorlayer:  %s is a ReconLayer" % self.name)

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
        mse = tf.reduce_mean(mse)                               # DH: theano: ((y - x) ** 2 ).sum(axis=1).mean()
            # mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y, x_recon)), reduction_indices = 1))
            # mse = tf.reduce_mean(tf.squared_difference(y, x_recon)) # DH: Error
            # mse = tf.sqrt(tf.reduce_mean(tf.square(y - x_recon)))   # DH: Error
        # Cross-entropy
        ce = cost.cross_entropy(y, x_recon)
            # ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x_recon))          # DH: list , list , Error (only be used for softmax output)
            # ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, x_recon))   # DH: list , index , Error (only be used for softmax output)
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[0]) \
                + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])           # faster than the code below
            # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))
        # DropNeuro
        P_o = cost.lo_regularizer(0.03)(self.train_params[0])   # + cost.lo_regularizer(0.5)(self.train_params[2])    # DH: if add lo on decoder, no neuron will be broken
        P_i = cost.li_regularizer(0.03)(self.train_params[0])  # + cost.li_regularizer(0.001)(self.train_params[2])
        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(activation_out)   # DH:  theano: T.mean( self.a[i] )                     # some neuron are broken, white and black
            # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=0) )         # DH: some neuron are broken, white and black
            # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=1) )   # DH: some neuron are broken, white and black
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
            # print('     use: mse, L2_w, KLD')
            # self.cost = mse + L2_w + KLD
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_o) can remove neurons of AE
            # print('     use: mse, L2_w, KLD, P_o')
            # self.cost = mse + L2_w + KLD + P_o
            # ----------------------------------------------------
            # Add DropNeuro penalty (P_i) can remove neurons of previous layer
            #   If previous layer is InputLayer, it means remove useless features
            print('     use: mse, L2_w, KLD, P_i')
            self.cost = mse + L2_w + KLD + P_i
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



# Dense+Noise layer
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
    ... Alternatively, you can choose a specific initializer for the weights as follow:
    ... network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1', W_init=tf.random_normal )
    >>> network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    >>> network = tl.layers.DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
    >>> network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    >>> network = tl.layers.DenseLayer(network, n_units=10, act = tl.activation.identity, name='output_layer')
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        name = 'dropout_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.n_units = layer.n_units
        print("  tensorlayer:Instantiate DropoutLayer %s: keep: %f" % (self.name, keep))

        set_keep[name] = tf.placeholder(tf.float32)
        self.outputs = tf.nn.dropout(self.inputs, set_keep[name])

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_drop.update( {set_keep[name]: keep} )
        self.all_layers.extend( [self.outputs] )
        # print(set_keep[name])    # Tensor("Placeholder_2:0", dtype=float32)
        # print(denoising1)           # Tensor("Placeholder_2:0", dtype=float32)
        # print(self.all_drop[denoising1])    # 0.8
        # exit()
        # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/tf/index.html
        # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
        # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
        # If the key is a SparseTensor, the value should be a SparseTensorValue.

class DropconnectDenseLayer(Layer):
    """
    The :class:`DropconnectDenseLayer` class is `DenseLayer` with DropConnect
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
        The arguments for the weights initializer.
    b_init_args : dictionary
        The arguments for the biases initializer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.8, n_units=800, act = tf.nn.relu, name='dropconnect_relu1')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.5, n_units=800, act = tf.nn.relu, name='dropconnect_relu2')
    >>> network = tl.layers.DropconnectDenseLayer(network, keep = 0.5, n_units=10, act = tl.activation.identity, name='output_layer')

    References
    ----------
    `Wan, L. (2013). Regularization of neural networks using dropconnect <http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13>`_
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        n_units = 100,
        act = tf.nn.relu,
        W_init = init.xavier_init,
        b_init = tf.zeros,
        W_init_args = {},
        b_init_args = {},
        name ='dropconnect_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_units
        print("  tensorlayer:Instantiate DropconnectDenseLayer %s: %d, %s" % (self.name, self.n_units, act))

        W = tf.Variable(W_init(shape=(n_in, n_units), **W_init_args), name='W')
        b = tf.Variable(b_init(shape=[n_units], **b_init_args), name='b')
        self.outputs = act(tf.matmul(self.inputs, W) + b)

        set_keep[name] = tf.placeholder(tf.float32)
        W_dropcon = tf.nn.dropout(W,  set_keep[name])
        self.outputs = act(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_drop.update( {set_keep[name]: keep} )
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )


# Convolutional layer
class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    n_units : int
        The number of units of the layer
    shape : list of shape
        shape of the filters
    strides : a list of ints. 1-D of length 4.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
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

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    >>>                   act = tf.nn.relu,
    >>>                   shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    >>>                   strides=[1, 1, 1, 1],
    >>>                   padding='SAME',
    >>>                   W_init = tf.truncated_normal,
    >>>                   W_init_args = {'mean' : 1, 'stddev':3},
    >>>                   b_init = tf.zeros,
    >>>                   b_init_args = {'name' : 'bias'},
    >>>                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> network = tl.layers.PoolLayer(network,
    ...                   ksize=[1, 2, 2, 1],
    ...                   strides=[1, 2, 2, 1],
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
        shape = [5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        W_init = tf.truncated_normal,
        b_init = tf.zeros,
        W_init_args = {},
        b_init_args = {},
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate Conv2dLayer %s: %s, %s, %s, %s" % (self.name, str(shape), str(strides), padding, act))

        W = tf.Variable(W_init(shape=shape, **W_init_args), name='W_conv')
        b = tf.Variable(b_init(shape=[shape[-1]], **b_init_args), name='b_conv')

        # W = tf.Variable( weights_initializer(shape=shape), name='W_conv')
        # b = tf.Variable( biases_initializer(shape=[shape[-1]]), name='b_conv')
        self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

class PoolLayer(Layer):
    """
    The :class:`PoolLayer` class is a 2D Pooling layer.

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
    see Conv2dLayer

    References
    ------------
    `TensorFlow Pooling <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`_
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
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate PoolLayer %s: %s, %s, %s, %s" % (self.name, str(ksize), str(strides), padding, pool))

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( [W] )


# Recurrent layer
class RNNLayer(Layer):
    """
    The :class:`RNNLayer` class is a RNN layer, you can implement vanilla RNN,
    LSTM and GRU with it.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    cell_fn : a TensorFlow's core RNN methods as follow.
        class tf.nn.rnn_cell.BasicRNNCell\n
        class tf.nn.rnn_cell.BasicLSTMCell\n
        class tf.nn.rnn_cell.GRUCell\n
        class tf.nn.rnn_cell.LSTMCell\n
        see `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_
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
    is_reshape : boolen
        Reshape the inputs to 3 dimension tensor.\n
        If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        If input is [batch_size * n_steps, n_features], we need to reshape it.
    name : a string or None
        An optional name to attach to this layer.

    Field (Class Variables)
    -----------------------
    outputs : a tensor
        The output of this RNN.
        return_last = False, outputs = all cell_output, which is the hidden state.
            cell_output.get_shape() = (?, n_hidden)
    state : a tensor
        When state_is_tuple = False,
        it is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden].\n
        When state_is_tuple = True, it stores two elements: (c, h), in that order.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, D])
    >>> network = tl.layers.InputLayer(x, name='input_layer')

    ... For single RNN
    >>> network = tl.layers.RNNLayer(x, n_hidden=200, n_steps=num_steps,
                    return_last=True, is_reshape=True, name='lstm_layer')
    >>> network = tl.layers.DenseLayer(network, n_units=n_classes,
    ...                 act = tl.activation.identity, name='output_layer')

    ...
    ... For multiple RNNs
    >>> network = tl.layers.RNNLayer(x, n_hidden=200, n_steps=num_steps,
    ...              return_last=False, is_reshape=True, name='lstm_layer1')
    >>> network = tl.layers.RNNLayer(network, n_hidden=200, n_steps=num_steps,
    ...              return_last=True, is_reshape=False, name='lstm_layer2')
    >>> network = tl.layers.DenseLayer(network, n_units=n_classes,
    ...                 act = tl.activation.identity, name='output_layer')


    Notes
    -----
    If the input to this layer has more than two axes, we need to flatten the
    input by using :class:`FlattenLayer`.

    References
    ----------
    `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_

    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.BasicRNNCell,
        n_hidden = 100,
        n_steps = 5,
        return_last = False,
        is_reshape = True,
        cell_init_args = {},
        name = 'rnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_hidden

        print("  tensorlayer:Instantiate RNNLayer %s: cell_fn:%s n_hidden:%d, n_steps:%d, dim:%d %s" % (self.name, cell_fn, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape()))

        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        if is_reshape:
            self.inputs = tf.reshape(self.inputs, shape=[-1, n_steps, int(self.inputs._shape[-1])])

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     batch_size of rnn : %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, use a tensor instead.")

        outputs = []
        cell = cell_fn(num_units=n_hidden, **cell_init_args)
        state = cell.zero_state(batch_size, dtype=tf.float32)
        with tf.variable_scope(name) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        # print('inputs', self.inputs)          # (?, 5, 6400)
        # print('state', state.get_shape())    # (?, 400)
        # for param in rnn_variables:
        #     print('rnn_var', param.get_shape(), param.name)    # (6600, 800)   (800,)
        # print('last output', outputs[-1].get_shape(), outputs[-1])
        # exit()

        # for p in tf.trainable_variables():
        #     print('all_var', p.get_shape(), p.name)

        if return_last:
            "Many to one"
            self.outputs = outputs[-1]
        else:
            "Many to many : Synced sequence input and output"
            self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, n_hidden])

        self.state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )



# Shape layer
class FlattenLayer(Layer):
    """
    The :class:`FlattenLayer` class is layer which reshape each high-dimension
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
        ''' Flatten the outputs to one dimension '''
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = flatten_reshape(self.inputs)
        self.n_units = int(self.outputs._shape[-1])
        print("  tensorlayer:Instantiate FlattenLayer %s, %d" % (self.name, self.n_units))
        self.all_layers = list(layer.all_layers)    # list() is pass by value (shallow), without list is pass by reference
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
        layer = None,
        name ='concat_layer',
    ):
        Layer.__init__(self, name=name)
        # self.inputs = layer.outputs
        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)
        # self.outputs = flatten_reshape(self.inputs)
        self.outputs = tf.concat(1, self.inputs)
        self.n_units = int(self.outputs._shape[-1])
        print("  tensorlayer:Instantiate ConcatLayer %s, %d" % (self.name, self.n_units))
        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)

        self.all_drop = dict(layer[0].all_drop)
        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))
        # self.all_drop = dict(layer.all_drop)
        # self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( [W, b] )

## Layers have not been tested
# dense
class MaxoutLayer(Layer):
    """
    Coming soon

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
        n_in = layer.n_units
        self.n_units = n_units
        print("  tensorlayer:Instantiate MaxoutLayer %s: %d" % (self.name, self.n_units))
        W = tf.Variable(init.xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
        b = tf.Variable(tf.zeros([n_units]), name='b')

        # self.outputs = act(tf.matmul(self.inputs, W) + b)
        # https://www.tensorflow.org/versions/r0.9/api_docs/python/array_ops.html#pack
        # http://stackoverflow.com/questions/34362193/how-to-explicitly-broadcast-a-tensor-to-match-anothers-shape-in-tensorflow
        # tf.concat tf.pack  tf.tile

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

# class ResnetLayer(Layer):
#     """
#     The :class:`ResnetLayer` class is a fully connected layer, while the inputs
#     are added on the outputs
#
#     Parameters
#     ----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     act : activation function
#         The function that is applied to the layer activations.
#     W_init : weights initializer
#         The initializer for initializing the weight matrix.
#     b_init : biases initializer
#         The initializer for initializing the bias vector.
#     W_init_args : dictionary
#         The arguments for the weights initializer.
#     b_init_args : dictionary
#         The arguments for the biases initializer.
#     name : a string or None
#         An optional name to attach to this layer.
#
#     Examples
#     --------
#     >>>
#
#     References
#     ----------
#     `He, K (2015) Deep Residual Learning for Image Recognition. <http://doi.org/10.3389/fpsyg.2013.00124>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         act = tf.nn.relu,
#         W_init = init.xavier_init,
#         b_init = tf.zeros,
#         W_init_args = {},
#         b_init_args = {},
#         name ='resnet_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         n_in = layer.n_units
#         self.n_units = n_in
#         print("  tensorlayer:Instantiate ResnetLayer %s: %d, %s" % (self.name, self.n_units, act))
#
#         W = tf.Variable(W_init(shape=(n_in, n_units), **W_init_args), name='W')
#         b = tf.Variable(b_init(shape=[n_units], **b_init_args), name='b')
#         self.outputs = act(tf.matmul(self.inputs, W) + b) + self.inputs
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( [W, b] )

# noise
class GaussianNoiseLayer(Layer):
    """
    Coming soon
    """
    def __init__(
        self,
        layer = None,
        # keep = 0.5,
        name = 'gaussian_noise_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.n_units = layer.n_units
        print("  tensorlayer:Instantiate GaussianNoiseLayer %s: keep: %f" % (self.name, keep))

# shape
class ReshapeLayer(Layer):
    """
    Deprecated and unused.
    """
    def __init__(
        self,
        layer = None,
        shape = None,
        name ='reshape_layer',
    ):
        raise Exception("Deprecated and unused.\n Hint : Use tf.reshape to reshape Layer.outputs instead.")
        self.n_units = layer.n_units
        print("  tensorlayer:Instantiate ReshapeLayer %s: %f" % (self.name, shape))

        self.outputs = tf.reshape(layer.outputs, shape=shape)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)


# rnn
class BidirectionalRNNLayer(Layer):
    """
    Coming soon

    The :class:`BidirectionalRNNLayer` class is a RNN layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_hidden : a int
        The number of hidden units in the layer.
    n_steps : a int
        The sequence length.
    return_last : boolen
        If True, return the last output, "Sequence input and single output"
        If False, return all outputs, "Synced sequence input and output"
        In other word, if you want to apply one or more RNN(s) on this layer, set to False.
    cell_init_args : a dictionary
        The arguments for the cell initializer.
    is_reshape : boolen
        Reshape the inputs to 3 dimension tensor.
        If input is［batch_size, n_steps, n_features], we do not need to reshape it.
        If input is [batch_size * n_steps, n_features], we need to reshape it.
    name : a string or None
        An optional name to attach to this layer.

    Field (Class Variables)
    -----------------------
    outputs : a tensor
        The output of this RNN.
    state : a tensor
        When state_is_tuple=False
        It is the final hidden and cell states, states.get_shape() = [?, 2 * n_hidden]

    Examples
    --------
    >>>

    Notes
    -----


    References
    ----------
    `Neural Network RNN Cells in TensorFlow <https://www.tensorflow.org/versions/master/api_docs/python/rnn_cell.html>`_

    """
    def __init__(
        self,
        layer = None,
        n_hidden = 100,
        n_steps = 5,
        return_last = False,
        is_reshape = True,
        cell_init_args = {'forget_bias': 1.0},#, 'input_size' : None, 'state_is_tuple' : False, 'activation' : 'tanh' },
        name = 'basic_lstm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_hidden

        print("  tensorlayer:Instantiate BidirectionalRNNLayer %s: n_hidden:%d, n_steps:%d, dim:%d %s" % (self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape()))


# cnn
class Conv3dLayer(Layer):
    """
    Coming soon

    The :class:`Conv3dLayer` class is a 3D CNN layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    n_units : int
        The number of units of the layer
    shape : list of shape
        shape of the filters
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

    Examples
    --------
    >>>
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
        shape = [5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        W_init = tf.truncated_normal,
        b_init = tf.zeros,
        W_init_args = {},
        b_init_args = {},
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate Conv3dLayer %s: %s, %s, %s, %s" % (self.name, str(shape), str(strides), padding, act))

        # W = tf.Variable(W_init(shape=shape, **W_init_args), name='W_conv')
        # b = tf.Variable(b_init(shape=[shape[-1]], **b_init_args), name='b_conv')
        #
        # self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )














#
