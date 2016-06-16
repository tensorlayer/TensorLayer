import tensorflow as tf
import time
import tensorlayer.init as init
import tensorlayer.visualize as visualize
import tensorlayer.utils as utils
import tensorlayer.cost as cost
import tensorlayer.iterate as iterate
import numpy as np

# __all__ = [
#     "Layer",
#     "DenseLayer",
# ]

## Dynamically creat variable for keep prob
# set_keep = locals()
set_keep = globals()

## Variable Operation
def flatten_reshape(variable):
    """
    The :function:`flatten_reshape` reshapes the input to a 1D vector.

    Parameters
    ----------
    variable : a tensorflow variable

    Examples
    --------
    >>> xxx
    >>> xxx
    """

    # ''' input a high-dimension variable, return a 1-D reshaped variable
    #     for example:
    #         W_conv2 = weight_variable([5, 5, 100, 32])   # 64 features for each 5x5 patch
    #         b_conv2 = bias_variable([32])
    #         W_fc1 = weight_variable([7 * 7 * 32, 256])
    #
    #         h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    #         h_pool2 = max_pool_2x2(h_conv2)
    #         h_pool2.get_shape()[:].as_list() = [batch_size, 7, 7, 32]
    #
    #         [batch_size, mask_row, mask_col, n_mask]
    #
    #         h_pool2_flat = tensorflatten(h_pool2)
    #         h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
    # '''
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
        if name in globals():
            raise Exception("Variable '%s' already exists, please choice other 'name'\nUse different name for different 'Layer'" % name)
        else:
            self.name = name

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
    inputs : a :tensorflow placeholder
        The input tensor data.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        inputs = None,
        name ='input_layer'
    ):
        Layer.__init__(self, inputs=inputs, name=name)
        # super(InputLayer, self).__init__()            # initialize all super classes
        self.n_units = int(inputs._shape[1])
        print("  tensorlayer:Instantiate InputLayer %s %s" % (self.name, inputs._shape))

        self.outputs = inputs

        self.all_layers = []
        self.all_params = []
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
        The number of units of the layer
    act : activation function
        The function that is applied to the layer activations.
    weights_initializer : weights initializer
        Initialize the weight matrix
    biases_initializer : biases initializer
        Initialize the bias vector
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> xxx
    >>> xxx

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
        weights_initializer = init.xavier_init,
        biases_initializer = tf.zeros,
        name ='dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_units
        print("  tensorlayer:Instantiate DenseLayer %s: %d, %s" % (self.name, self.n_units, act))
        # self.shape = inputs._shape
        # print(vars(self))
        # exit()
            # W = tf.get_variable("W", shape=[n_in, n_units], initializer=tf.contrib.layers.xavier_initializer())
        # if act == tf.nn.relu:
        #     W = tf.Variable(tf.random_uniform([n_in, n_units], minval=0, maxval=0.01, dtype=tf.float32, seed=None, name=None), name='W')
        # else:
        #     W = tf.Variable(tf.random_normal([n_in, n_units], stddev=0.01), name='W')
        W = tf.Variable(weights_initializer(shape=(n_in, n_units)), name='W')
        # W = tf.Variable(init.xavier_init(shape=(n_in, n_units), uniform=True), name='W')
        # W = tf.Variable(tf.constant(.01, shape=[n_in, n_units]), name='W')
        b = tf.Variable(biases_initializer([n_units]), name='b')
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
        The data used for reconstruct
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
    >>> xxx
    >>> xxx

    Notes
    -----
    The input layer should be `DenseLayer` or a layer has only one axes.
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

        lambda_l2_w = 0.004
        learning_rate = 0.0001 * 1
        print("     lambda_l2_w: %f" % lambda_l2_w)
        print("     learning_rate: %f" % learning_rate)

        # print(vars(x_recon))
        # exit()

        y = self.outputs
        self.train_params = self.all_params[-4:]

        '''
        You may want to modify this part to define your own cost function

        by default, the cost is implemented as:
         for sigmoid layer, the implementation follow:
             Ref:  http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
         for rectifying layer, the implementation follow:
             Ref: Glorot, X., Bordes, A., & Bengio, Y. (2011).
                  Deep Sparse Rectifier Neural Networks. Aistats,
                  15, 315–323. http://doi.org/10.1.1.208.6449
        '''

        # mean-squre-error = quadratic-cost
        mse = tf.reduce_sum(tf.squared_difference(y, x_recon), reduction_indices = 1)
        mse = tf.reduce_mean(mse)                               # theano: ((y - x) ** 2 ).sum(axis=1).mean()
            # mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y, x_recon)), reduction_indices = 1))
            # mse = tf.reduce_mean(tf.squared_difference(y, x_recon)) # Error
            # mse = tf.sqrt(tf.reduce_mean(tf.square(y - x_recon)))   # Error
        # cross-entropy
        ce = cost.cross_entropy(y, x_recon)
            # ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x_recon))          # list , list , Error (only be used for softmax output)
            # ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, x_recon))   # list , index , Error (only be used for softmax output)
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[0]) \
                + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])           # faster than the code below
            # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))
        # DropNeuro
        P_o = cost.lo_regularizer(0.001)(self.train_params[0]) + cost.lo_regularizer(0.001)(self.train_params[2])
        P_i = cost.li_regularizer(0.001)(self.train_params[0]) + cost.li_regularizer(0.001)(self.train_params[2])

        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(activation_out)   # theano: T.mean( self.a[i] )                     # some neuron are broken, white and black
            # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=0) )         # some neuron are broken, white and black
            # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, reduction_indices=1) )   # some neuron are broken, white and black
        # KLD
        beta = 4
        rho = 0.15
        p_hat = tf.reduce_mean(activation_out, reduction_indices = 0)   # theano: p_hat = T.mean( self.a[i], axis=0 )
        KLD = beta * tf.reduce_sum( rho * tf.log(tf.div(rho, p_hat)) + (1- rho) * tf.log((1- rho)/ (tf.sub(float(1), p_hat))) )
            # KLD = beta * tf.reduce_sum( rho * tf.log(rho/ p_hat) + (1- rho) * tf.log((1- rho)/(1- p_hat)) )
            # theano: L1_a = l1_a[i] * T.sum( rho[i] * T.log(rho[i]/ p_hat) + (1- rho[i]) * T.log((1- rho[i])/(1- p_hat)) )

        if act == tf.nn.softplus:
            print('     use: mse, L2_w, L1_a')
            self.cost = mse + L1_a + L2_w #P_o
        elif act == tf.nn.sigmoid:
            print('     use: ce, L2_w, KLD')
            self.cost = ce + L2_w + KLD
                # self.cost = mse + L2_w + KLD
        else:
            raise Exception("Don't support the given reconstruct activation function")

        self.train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08, use_locking=False).minimize(self.cost, var_list=self.train_params)
                # self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost, var_list=self.train_params)

        # You may want to check different cost values
        # self.KLD = KLD
        # self.L2_w = L2_w
        # self.mse = mse
        # self.L1_a = L1_a

    def pretrain(self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10,
                  save=True, save_name='w1pre_'):
        print("     tensorlayer:  %s start pretrain" % self.name)
        print("     batch_size: %d" % batch_size)
        if denoise_name:
            print("     denoising layer keep: %f" % self.all_drop[set_keep[denoise_name]])
            dp_denoise = self.all_drop[set_keep[denoise_name]]
        else:
            print("     no denoising layer")

        # You may want to check different cost values
        # dp_dict = utils.dict_to_one( self.all_drop )
        # feed_dict = {x: X_val}
        # feed_dict.update(dp_dict)
        # print(sess.run([self.mse, self.L1_a, self.L2_w], feed_dict=feed_dict))
        # exit()

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
                        visualize.W(self.train_params[0].eval(), second=10, saveable=True, name=save_name+str(epoch+1), fig_idx=2012)
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
    >>> xxx
    >>> xxx
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
        The number of units of the layer
    act : activation function
        The function that is applied to the layer activations.
    weights_initializer : weights initializer
        Initialize the weight matrix
    biases_initializer : biases initializer
        Initialize the bias vector
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> xxx
    >>> xxx
    """
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        n_units = 100,
        act = tf.nn.relu,
        weights_initializer = init.xavier_init,
        biases_initializer = tf.zeros,
        name ='dropconnect_layer',
    ):
        '''
        Single DenseLayer with dropconnect behaviour

        Wan, L., Zeiler, M., Zhang, S., LeCun, Y., & Fergus, R. (2013).
        Regularization of neural networks using dropconnect. Icml, (1), 109–111.
        Retrieved from http://machinelearning.wustl.edu/mlpapers/papers/icml2013_wan13
        '''
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_units
        print("  tensorlayer:Instantiate DropconnectDenseLayer %s: %d, %s" % (self.name, self.n_units, act))

        # W = tf.Variable(init.xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
        # b = tf.Variable(tf.zeros([n_units]), name='b')
        W = tf.Variable(weights_initializer(shape=(n_in, n_units)), name='W')
        b = tf.Variable(biases_initializer([n_units]), name='b')
        set_keep[name] = tf.placeholder(tf.float32)
        W_dropcon = tf.nn.dropout(W,  set_keep[name])
        self.outputs = act(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_drop.update( {set_keep[name]: keep} )
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

# Convolutional Layer
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
        XXX
    strides: list of stride
        XXX
    padding: a string
        XXX
    weights_initializer : weights initializer
        Initialize the weight matrix
    biases_initializer : biases initializer
        Initialize the bias vector
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> xxx
    >>> xxx
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
        shape = [5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        weights_initializer = tf.truncated_normal,
        biases_initializer = tf.zeros,
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate Conv2dLayer %s: %s, %s, %s, %s" % (self.name, str(shape), str(strides), padding, act))

        # W = tf.Variable( tf.truncated_normal(shape=shape, stddev=0.1, seed=np.random.randint(99999999)), name='W_conv')
        # b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name='b_conv')
        W = tf.Variable( weights_initializer(shape=shape), name='W_conv')
        b = tf.Variable( biases_initializer(shape=[shape[-1]]), name='b_conv')
        self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

class Pool2dLayer(Layer):
    """
    The :class:`Pool2dLayer` class is a 2D Pooling layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    ksize : list of XX
        XXX
    strides: list of stride
        XXX
    padding: a string
        XXX
    pool: a pooling function
        tf.nn.max_pool ...
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> xxx
    >>> xxx
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
        '''
        https://www.tensorflow.org/versions/r0.9/api_docs/python/nn.html#pooling
        '''
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate Pool2dLayer %s: %s, %s, %s, %s" % (self.name, str(ksize), str(strides), padding, pool))

        self.outputs = pool(self.inputs, ksize=ksize, strides=strides, padding=padding)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        # self.all_params.extend( [W] )

# Shape layer
class FlattenLayer(Layer):
    """
    The :class:`FlattenLayer` class is layer which reshape the input to a 1D
    vector.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> xxx
    >>> xxx
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

# Merge layer
    # ConcatLayer

## Layers have not been tested yet
# dense
class MaxoutLayer(Layer):
    """
    Coming soon
    """
    def __init__(
        self,
        layer = None,
        n_units = 100,
        name ='maxout_layer',
    ):
        '''
        Single DenseLayer with Max-out behaviour, work well with DropOut

        Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013).
        Maxout Networks. arXiv Preprint, 1319–1327. Retrieved from http://arxiv.org/abs/1302.4389
        '''
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
# densen
class ResnetLayer(Layer):
    """
    Coming soon
    """
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
        name ='resnet_layer',
    ):
        '''
        Single DenseLayer, while the inputs are added on the outputs

        He, K., Zhang, X., Ren, S., & Sun, J. (2015).
        Deep Residual Learning for Image Recognition.
        Arxiv.Org, 7(3), 171–180. http://doi.org/10.3389/fpsyg.2013.00124
        '''
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        n_in = layer.n_units
        self.n_units = n_in
        print("  tensorlayer:Instantiate ResnetLayer %s: %d, %s" % (self.name, self.n_units, act))
        W = tf.Variable(init.xavier_init(n_inputs=n_in, n_outputs=self.n_units, uniform=True), name='W')
        b = tf.Variable(tf.zeros([self.n_units]), name='b')

        self.outputs = act(tf.matmul(self.inputs, W) + b) + self.inputs

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )
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
    Coming soon
    """
    def __init__(
        self,
        layer = None,
        shape = None,
        name ='reshape_layer',
    ):
        pass
# merge
class ConcatLayer(Layer):
    """
    Coming soon
    """
    def __init__(
        self,
        layer = None,
        name ='concat_layer',
    ):
        pass
















#
