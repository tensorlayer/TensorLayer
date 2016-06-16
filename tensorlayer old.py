#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from sys import platform as _platform
# import copy

# Variable class: https://www.tensorflow.org/versions/r0.8/api_docs/python/state_ops.html#Variable
# tf.contrib.layers.apply_regularization:  https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#apply_regularization
# Layers (contrib): Higher level ops for building neural network layers.
#                        https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#layers-contrib
#       Regularizers:    https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#l1_regularizer

# np.random.seed(0)
# tf.set_random_seed(0)

## Dynamically creat variable for keep prob
set_keep = locals()
# set_keep = globals()

## System
def exit_tf(sess):
    text = "Close tensorboard and nvidia-process if available"
    sess.close()
    # import time
    # time.sleep(2)
    if _platform == "linux" or _platform == "linux2":
        print('linux: %s' % text)
        os.system('nvidia-smi')
        os.system('fuser 6006/tcp -k')  # kill tensorboard 6006
        os.system("nvidia-smi | grep python |awk '{print $3}'|xargs kill") # kill all nvidia-smi python process
    elif _platform == "darwin":
        print('OS X: %s' % text)
        os.system("lsof -i tcp:6006 | grep -v PID | awk '{print $2}' | xargs kill") # kill tensorboard 6006
    elif _platform == "win32":
        print('Windows: %s' % text)
    else:
        print(_platform)
    exit()

def clear_all(printable=True):
    """Clears all the placeholder variables from of the application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue
        if 'class' in str(globals()[var]): continue

        if printable:
            print(" clear_all ------- %s" % str(globals()[var]))

        del globals()[var]

def set_gpu_fraction(gpu_fraction=0.3):
    print("  tensorlayer: GPU MEM Fraction %f" % gpu_fraction)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
    return sess

## Visualization
def visualize_W(W, second=10, saveable=True, name='mnist', fig_idx=2396512):
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images
    size = W.shape[0]
    n_units = W.shape[1]

    num_r = int(np.sqrt(n_units))  # 每行显示的个数   若25个hidden unit -> 每行显示5个
    num_c = int(np.ceil(n_units/num_r))
    count = int(1)
    for row in range(1, num_r+1):
        for col in range(1, num_c+1):
            if count > n_units:
                break
            a = fig.add_subplot(num_r, num_c, count)
            # plt.imshow(np.reshape(W.get_value()[:,count-1],(28,28)), cmap='gray')
            plt.imshow(np.reshape(W[:,count-1] / np.sqrt( (W[:,count-1]**2).sum()) ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
            # plt.imshow(np.reshape(W[:,count-1] ,(np.sqrt(size),np.sqrt(size))), cmap='gray', interpolation="nearest")
            plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            count = count + 1
    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

def visualize_frame(I, second=5, saveable=True, name='frame', fig_idx=12836):
    ''' display a frame. Make sure OpenAI Gym render() is disable before using it. '''
    if saveable is False:
        plt.ion()
    fig = plt.figure(fig_idx)      # show all feature images

    plt.imshow(I)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if saveable:
        plt.savefig(name+'.pdf',format='pdf')
    else:
        plt.draw()
        plt.pause(second)

## Cost Functions
def cross_entropy(output, target):
  """Cross entropy loss
  See https://en.wikipedia.org/wiki/Cross_entropy
      https://github.com/cmgreen210/TensorFlowDeepAutoencoder/blob/master/code/ae/autoencoder.py
  Args:
    output: tensor of net output
    target: tensor of net we are trying to reconstruct
  Returns:
    Scalar tensor of cross entropy
  """
  with tf.name_scope("cross_entropy_loss"):
      net_output_tf = output
      target_tf = target
      cross_entropy = tf.add(tf.mul(tf.log(net_output_tf, name=None),target_tf),
                             tf.mul(tf.log(1 - net_output_tf), (1 - target_tf)))
      return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1), name='cross_entropy_mean')

def li_regularizer(scale):
  """Returns a function that can be used to apply group li regularization to weights.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  li regularization removes the neurons of previous layer.
  'i' represents 'inputs'

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `li(weights, name=None)` that apply L1
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops
  # from tensorflow.python.platform import tf_logging as logging

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def li(weights, name=None):
    """Applies li regularization to weights."""
    with ops.op_scope([weights], name, 'li_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
    #   return standard_ops.mul(
    #       my_scale,
    #       standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(weights**2, 1))),
    #       name=scope)
    return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
        #   standard_ops.reduce_mean(standard_ops.sqrt(standard_ops.reduce_mean(tf.square(weights), 1))),
          name=scope)
  return li

def lo_regularizer(scale):
  """Returns a function that can be used to apply group lo regularization to weights.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  Lo regularization removes the neurons of current layer.
  'o' represents outputs

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `lo(weights, name=None)` that apply Lo
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops
  # from tensorflow.python.platform import tf_logging as logging

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale >= 1.:
      raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def lo(weights, name=None):
    """Applies group column regularization to weights."""
    with ops.op_scope([weights], name, 'lo_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
    #   return standard_ops.mul(
    #       my_scale,
    #       standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(weights**2, 0))),
    #       name=scope)
      return standard_ops.mul(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 0))),
        #   standard_ops.reduce_mean(standard_ops.sqrt(standard_ops.reduce_mean(tf.square(weights), 0))),
          name=scope)
  return lo

def maxnorm_regularizer(scale=1.0):
  """Returns a function that can be used to apply max-norm regularization to weights.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
  Max-norm regularization

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `lo(weights, name=None)` that apply Lo
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    # if scale >= 1.:
    #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
    #                    scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def mn(weights, name=None):
    """Applies max-norm regularization to weights."""
    with ops.op_scope([weights], name, 'maxnorm_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.mul(my_scale, standard_ops.reduce_max(standard_ops.abs(weights)), name=scope)
  return mn

def maxnorm_o_regularizer(scale):
  """Returns a function that can be used to apply max-norm regularization to weights.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/regularizers.py
  https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
  Max-norm output regularization removes the neurons of current layer.

  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.

  Returns:
    A function with signature `lo(weights, name=None)` that apply Lo
    regularization.

  Raises:
    ValueError: If scale is outside of the range [0.0, 1.0] or if scale is not a
    float.
  """
  import numbers
  from tensorflow.python.framework import ops
  from tensorflow.python.ops import standard_ops

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    # if scale >= 1.:
    #   raise ValueError('Setting a scale greater than 1 on a regularizer: %g' %
    #                    scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _, name=None: None

  def mn_o(weights, name=None):
    """Applies max-norm regularization to weights."""
    with ops.op_scope([weights], name, 'maxnorm_o_regularizer') as scope:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                               name='scale')
      return standard_ops.mul(my_scale, standard_ops.reduce_sum(standard_ops.reduce_max(standard_ops.abs(weights), 0)), name=scope)
  return mn_o

## Cost Functions have not been tested yet


## Parameters Initialization
def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  import math
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    # return tf.random_uniform_initializer(-init_range, init_range)
    return tf.random_uniform(shape=[n_inputs, n_outputs], minval=-init_range, maxval=init_range, dtype=tf.float32, seed=None, name=None)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    # return tf.truncated_normal_initializer(stddev=stddev)
    return tf.truncated_normal(shape=[n_input, n_outputs], mean=0.0, stddev=stddev, dtype=tf.float32, seed=None, name=None)

## Load Data - file
def load_mnist_dataset(shape=(-1,784)):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(shape)
        # data = data.reshape(-1, 1, 28, 28)    # for lasagne
        # data = data.reshape(-1, 28, 28, 1)      # for tensorflow
        # data = data.reshape(-1, 784)      # for tensorflow
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    ## you may want to change the path
    data_dir = ''   #os.getcwd() + '/lasagne_tutorial/'
    # print('data_dir > %s' % data_dir)

    X_train = load_mnist_images(data_dir+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_dir+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    ## you may want to plot one example
    # print('X_train[0][0] >', X_train[0][0].shape, type(X_train[0][0]))  # for lasagne
    # print('X_train[0] >', X_train[0].shape, type(X_train[0]))       # for tensorflow
    # # exit()
    #         #  [[..],[..]]      (28, 28)      numpy.ndarray
    #         # plt.imshow 只支持 (28, 28)格式，不支持 (1, 28, 28),所以用 [0][0]
    # fig = plt.figure()
    # #plotwindow = fig.add_subplot(111)
    # # plt.imshow(X_train[0][0], cmap='gray')    # for lasagne (-1, 1, 28, 28)
    # plt.imshow(X_train[0].reshape(28,28), cmap='gray')     # for tensorflow (-1, 28, 28, 1)
    # plt.title('A training image')
    # plt.show()

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

## Evaluation
def evaluation(y_test, y_predict, n_classes):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    c_mat = confusion_matrix(y_test, y_predict, labels = [x for x in range(n_classes)])
    f1    = f1_score(y_test, y_predict, average = None, labels = [x for x in range(n_classes)])
    f1_macro = f1_score(y_test, y_predict, average='macro')
    acc   = accuracy_score(y_test, y_predict)
    print('confusion matrix: \n',c_mat)
    print('f1-score:',f1)
    print('f1-score(macro):',f1_macro)   # same output with > f1_score(y_true, y_pred, average='macro')
    print('accuracy-score:', acc)
    return c_mat, f1, acc, f1_macro

## Iteration
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle: # 打乱顺序
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

## Activation
def identity(x):
    ''' identity activation function '''
    return x

def ramp(x, v_min=0, v_max=1, name=None):
    ''' ramp activation function '''
    return tf.clip_by_value(x, clip_value_min=v_min, clip_value_max=v_max, name=name)

## Variable Operation
def flatten_reshape(variable):
    ''' input a high-dimension variable, return a 1-D reshaped variable
        for example:
            W_conv2 = weight_variable([5, 5, 100, 32])   # 64 features for each 5x5 patch
            b_conv2 = bias_variable([32])
            W_fc1 = weight_variable([7 * 7 * 32, 256])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            h_pool2.get_shape()[:].as_list() = [batch_size, 7, 7, 32]

            [batch_size, mask_row, mask_col, n_mask]

            h_pool2_flat = tensorflatten(h_pool2)
            h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)
    '''
    dim = 1
    for d in variable.get_shape()[1:].as_list():
        dim *= d
    return tf.reshape(variable, shape=[-1, dim])

## Layers
# Layer base class
class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
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

    @staticmethod
    def dict_to_one(dp_dict):
        ''' input a dictionary, return a dictionary that all items are set to one
            use for disable dropout layer.
                dp_dict = Layer.dict_to_one( network.all_drop )
                feed_dict = {x: X_val, y_: y_val}
                feed_dict.update(dp_dict)
        '''
        return {x: 1 for x in dp_dict}

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

# Network input
class InputLayer(Layer):
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
    tensorlayer.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)
    A fully connected layer.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    Examples
    --------
    >>> xxx
    >>> xxx
    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.nn.relu,
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
        W = tf.Variable(xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
        # W = tf.Variable(tf.constant(.01, shape=[n_in, n_units]), name='W')
        b = tf.Variable(tf.zeros([n_units]), name='b')
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
        ce = cross_entropy(y, x_recon)
            # ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x_recon))          # list , list , Error (only be used for softmax output)
            # ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, x_recon))   # list , index , Error (only be used for softmax output)
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[0]) \
                + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])           # faster than the code below
            # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))
        # DropNeuro
        P_o = lo_regularizer(0.001)(self.train_params[0]) + lo_regularizer(0.001)(self.train_params[2])
        P_i = li_regularizer(0.001)(self.train_params[0]) + li_regularizer(0.001)(self.train_params[2])

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
        # dp_dict = Layer.dict_to_one( self.all_drop )
        # feed_dict = {x: X_val}
        # feed_dict.update(dp_dict)
        # print(sess.run([self.mse, self.L1_a, self.L2_w], feed_dict=feed_dict))
        # exit()

        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, _ in iterate_minibatches(X_train, X_train, batch_size, shuffle=True):
                dp_dict = Layer.dict_to_one( self.all_drop )
                if denoise_name:
                    dp_dict[set_keep[denoise_name]] = dp_denoise
                feed_dict = {x: X_train_a}
                feed_dict.update(dp_dict)
                sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, n_batch = 0, 0
                for X_train_a, _ in iterate_minibatches(X_train, X_train, batch_size, shuffle=True):
                    dp_dict = Layer.dict_to_one( self.all_drop )
                    feed_dict = {x: X_train_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch += 1
                print("   train loss: %f" % (train_loss/ n_batch))
                val_loss, n_batch = 0, 0
                for X_val_a, _ in iterate_minibatches(X_val, X_val, batch_size, shuffle=True):
                    dp_dict = Layer.dict_to_one( self.all_drop )
                    feed_dict = {x: X_val_a}
                    feed_dict.update(dp_dict)
                    err = sess.run(self.cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch += 1
                print("   val loss: %f" % (val_loss/ n_batch))
                if save:
                    try:
                        visualize_W(self.train_params[0].eval(), second=10, saveable=True, name=save_name+str(epoch+1), fig_idx=2012)
                    except:
                        raise Exception("You should change visualize_W(), if you want to save the feature images for different dataset")

# Noise layer
class DropoutLayer(Layer):
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
    def __init__(
        self,
        layer = None,
        keep = 0.5,
        n_units = 100,
        act = tf.nn.relu,
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

        W = tf.Variable(xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
        b = tf.Variable(tf.zeros([n_units]), name='b')
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
    def __init__(
        self,
        layer = None,
        act = tf.nn.relu,
        shape = [5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name ='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # n_in = layer.n_units
        print("  tensorlayer:Instantiate Conv2dLayer %s: %s, %s, %s, %s" % (self.name, str(shape), str(strides), padding, act))

        W = tf.Variable( tf.truncated_normal(shape=shape, stddev=0.1, seed=np.random.randint(99999999)), name='W_conv')
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name='b_conv')
        self.outputs = act( tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding) + b )

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )

# Pooling layer
class Pool2dLayer(Layer):
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

  # modules/layers
  # modules/updates
  # modules/init
  # modules/nonlinearities
  # modules/objectives
  # modules/regularization
  # modules/random
  # modules/utils

## Layers have not been tested yet
# dense
class MaxoutLayer(Layer):
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
        W = tf.Variable(xavier_init(n_inputs=n_in, n_outputs=n_units, uniform=True), name='W')
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
        W = tf.Variable(xavier_init(n_inputs=n_in, n_outputs=self.n_units, uniform=True), name='W')
        b = tf.Variable(tf.zeros([self.n_units]), name='b')

        self.outputs = act(tf.matmul(self.inputs, W) + b) + self.inputs

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( [W, b] )
# noise
class GaussianNoiseLayer(Layer):
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
    def __init__(
        self,
        layer = None,
        shape = None,
        name ='reshape_layer',
    ):
        pass
# merge
class ConcatLayer(Layer):
    def __init__(
        self,
        layer = None,
        name ='concat_layer',
    ):
        pass

## Testing Scripts
# def main_test_layers(model='relu'):
#     X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset(shape=(-1,784))
#
#     X_train = np.asarray(X_train, dtype=np.float32)
#     y_train = np.asarray(y_train, dtype=np.int64)
#     X_val = np.asarray(X_val, dtype=np.float32)
#     y_val = np.asarray(y_val, dtype=np.int64)
#     X_test = np.asarray(X_test, dtype=np.float32)
#     y_test = np.asarray(y_test, dtype=np.int64)
#
#     print('X_train.shape', X_train.shape)
#     print('y_train.shape', y_train.shape)
#     print('X_val.shape', X_val.shape)
#     print('y_val.shape', y_val.shape)
#     print('X_test.shape', X_test.shape)
#     print('y_test.shape', y_test.shape)
#     print('X %s   y %s' % (X_test.dtype, y_test.dtype))
#
#     sess = tf.InteractiveSession()
#
#     # placeholder
#     x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
#     y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
#
#     if model == 'relu':
#         network = InputLayer(x, name='input_layer')
#         network = DropoutLayer(network, keep=0.8, name='drop1')
#         network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu1')
#         network = DropoutLayer(network, keep=0.5, name='drop2')
#         network = DenseLayer(network, n_units=800, act = tf.nn.relu, name='relu2')
#         network = DropoutLayer(network, keep=0.5, name='drop3')
#         network = DenseLayer(network, n_units=10, act = identity, name='output_layer')
#     elif model == 'resnet':
#         network = InputLayer(x, name='input_layer')
#         network = DropoutLayer(network, keep=0.8, name='drop1')
#         network = ResnetLayer(network, act = tf.nn.relu, name='resnet1')
#         network = DropoutLayer(network, keep=0.5, name='drop2')
#         network = ResnetLayer(network, act = tf.nn.relu, name='resnet2')
#         network = DropoutLayer(network, keep=0.5, name='drop3')
#         network = DenseLayer(network, act = identity, name='output_layer')
#     elif model == 'dropconnect':
#         network = InputLayer(x, name='input_layer')
#         network = DropconnectDenseLayer(network, keep = 0.8, n_units=800, act = tf.nn.relu, name='dropconnect_relu1')
#         network = DropconnectDenseLayer(network, keep = 0.5, n_units=800, act = tf.nn.relu, name='dropconnect_relu2')
#         network = DropconnectDenseLayer(network, keep = 0.5, n_units=10, act = identity, name='output_layer')
#
#     # attrs = vars(network)
#     # print(', '.join("%s: %s\n" % item for item in attrs.items()))
#
#     # print(network.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}
#     # print(drop1, drop2, drop3)  # Tensor("Placeholder_2:0", dtype=float32) Tensor("Placeholder_3:0", dtype=float32) Tensor("Placeholder_4:0", dtype=float32)
#     # exit()
#
#     y = network.outputs
#     y_op = tf.argmax(tf.nn.softmax(y), 1)
#     ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
#     cost = ce
#
#     # cost = cost + maxnorm_regularizer(1.0)(network.all_params[0]) + maxnorm_regularizer(1.0)(network.all_params[2])
#     # cost = cost + lo_regularizer(0.0001)(network.all_params[0]) + lo_regularizer(0.0001)(network.all_params[2])
#     cost = cost + maxnorm_o_regularizer(0.001)(network.all_params[0]) + maxnorm_o_regularizer(0.001)(network.all_params[2])
#
#
#     params = network.all_params
#     # train
#     n_epoch = 500
#     batch_size = 128
#     learning_rate = 0.0001
#     print_freq = 10
#     # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
#     train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)
#
#     sess.run(tf.initialize_all_variables()) # initialize all variables
#
#     network.print_params()
#     network.print_layers()
#
#     print('   learning_rate: %f' % learning_rate)
#     print('   batch_size: %d' % batch_size)
#
#     for epoch in range(n_epoch):
#         start_time = time.time()
#         for X_train_a, y_train_a in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
#             feed_dict = {x: X_train_a, y_: y_train_a}
#             feed_dict.update( network.all_drop )    # enable all dropout/dropconnect/denoising layers
#             sess.run(train_op, feed_dict=feed_dict)
#
#             # The optional feed_dict argument allows the caller to override the value of tensors in the graph. Each key in feed_dict can be one of the following types:
#             # If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder, the shape of the value will be checked for compatibility with the placeholder.
#             # If the key is a SparseTensor, the value should be a SparseTensorValue.
#
#         if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#             print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#             dp_dict = Layer.dict_to_one( network.all_drop ) # disable all dropout/dropconnect/denoising layers
#             feed_dict = {x: X_train, y_: y_train}
#             feed_dict.update(dp_dict)
#             print("   train loss: %f" % sess.run(cost, feed_dict=feed_dict))
#             dp_dict = Layer.dict_to_one( network.all_drop )
#             feed_dict = {x: X_val, y_: y_val}
#             feed_dict.update(dp_dict)
#             print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
#             print("   val acc: %f" % np.mean(y_val == sess.run(y_op, feed_dict=feed_dict)))
#             try:
#                 visualize_W(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
#             except:
#                 raise Exception("You should change visualize_W(), if you want to save the feature images for different dataset")
#
#     print('Evaluation')
#     dp_dict = Layer.dict_to_one( network.all_drop )
#     feed_dict = {x: X_test, y_: y_test}
#     feed_dict.update(dp_dict)
#     print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
#     print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))
#
#     # Add ops to save and restore all the variables.
#     # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
#     saver = tf.train.Saver()
#     # you may want to save the model
#     save_path = saver.save(sess, "model.ckpt")
#     print("Model saved in file: %s" % save_path)
#     sess.close()
#
# def main_test_denoise_AE(model='relu'):
#     X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset(shape=(-1,784))
#
#     X_train = np.asarray(X_train, dtype=np.float32)
#     y_train = np.asarray(y_train, dtype=np.int64)
#     X_val = np.asarray(X_val, dtype=np.float32)
#     y_val = np.asarray(y_val, dtype=np.int64)
#     X_test = np.asarray(X_test, dtype=np.float32)
#     y_test = np.asarray(y_test, dtype=np.int64)
#
#     print('X_train.shape', X_train.shape)
#     print('y_train.shape', y_train.shape)
#     print('X_val.shape', X_val.shape)
#     print('y_val.shape', y_val.shape)
#     print('X_test.shape', X_test.shape)
#     print('y_test.shape', y_test.shape)
#     print('X %s   y %s' % (X_test.dtype, y_test.dtype))
#
#     sess = tf.InteractiveSession()
#
#     # placeholder
#     x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
#     y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
#
#     print("Build Network")
#     if model == 'relu':
#         network = InputLayer(x, name='input_layer')
#         network = DropoutLayer(network, keep=0.5, name='denoising1')    # if drop some inputs, it is denoise AE
#         network = DenseLayer(network, n_units=196, act = tf.nn.relu, name='relu1')
#         recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act = tf.nn.softplus, name='recon_layer1')
#     elif model == 'sigmoid':
#         # sigmoid - set keep to 1.0, if you want a vanilla Autoencoder
#         network = InputLayer(x, name='input_layer')
#         network = DropoutLayer(network, keep=0.5, name='denoising1')
#         network = DenseLayer(network, n_units=200, act=tf.nn.sigmoid, name='sigmoid1')
#         recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')
#
#     ## ready to train
#     sess.run(tf.initialize_all_variables())
#
#     ## print all params
#     print("All Network Params")
#     network.print_params()
#
#     ## pretrain
#     print("Pre-train Layer 1")
#     recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
#         # recon_layer1.pretrain(sess, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=1000, batch_size=128, print_freq=10)
#
#     # Add ops to save and restore all the variables.
#     # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
#     saver = tf.train.Saver()
#     # you may want to save the model
#     save_path = saver.save(sess, "model.ckpt")
#     print("Model saved in file: %s" % save_path)
#     sess.close()
#
# def main_test_stacked_denoise_AE(model='relu'):
#     # Load MNIST dataset
#     X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset(shape=(-1,784))
#
#     X_train = np.asarray(X_train, dtype=np.float32)
#     y_train = np.asarray(y_train, dtype=np.int64)
#     X_val = np.asarray(X_val, dtype=np.float32)
#     y_val = np.asarray(y_val, dtype=np.int64)
#     X_test = np.asarray(X_test, dtype=np.float32)
#     y_test = np.asarray(y_test, dtype=np.int64)
#
#     print('X_train.shape', X_train.shape)
#     print('y_train.shape', y_train.shape)
#     print('X_val.shape', X_val.shape)
#     print('y_val.shape', y_val.shape)
#     print('X_test.shape', X_test.shape)
#     print('y_test.shape', y_test.shape)
#     print('X %s   y %s' % (X_test.dtype, y_test.dtype))
#
#     sess = tf.InteractiveSession()
#
#     x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
#     y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
#
#     if model == 'relu':
#         act = tf.nn.relu
#         act_recon = tf.nn.softplus
#     elif model == 'sigmoid':
#         act = tf.nn.sigmoid
#         act_recon = act
#
#     # Define network
#     print("\nBuild Network")
#     network = InputLayer(x, name='input_layer')
#     # denoise layer for AE
#     network = DropoutLayer(network, keep=0.5, name='denoising1')
#     # 1st layer
#     network = DropoutLayer(network, keep=0.8, name='drop1')
#     network = DenseLayer(network, n_units=800, act = act, name=model+'1')
#     x_recon1 = network.outputs
#     recon_layer1 = ReconLayer(network, x_recon=x, n_units=784, act = act_recon, name='recon_layer1')
#     # 2nd layer
#     network = DropoutLayer(network, keep=0.5, name='drop2')
#     network = DenseLayer(network, n_units=800, act = act, name=model+'2')
#     recon_layer2 = ReconLayer(network, x_recon=x_recon1, n_units=800, act = act_recon, name='recon_layer2')
#     # 3rd layer
#     network = DropoutLayer(network, keep=0.5, name='drop3')
#     network = DenseLayer(network, n_units=10, act = identity, name='output_layer')
#
#     # Define fine-tune process
#     y = network.outputs
#     y_op = tf.argmax(tf.nn.softmax(y), 1)
#     ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
#     cost = ce
#
#     n_epoch = 500
#     batch_size = 128
#     learning_rate = 0.0001
#     print_freq = 10
#
#     train_params = network.all_params
#
#         # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
#     train_op = tf.train.AdamOptimizer(learning_rate , beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
#
#     # Initialize all variables including weights, biases and the variables in train_op
#     sess.run(tf.initialize_all_variables())
#
#     # Pre-train
#     print("\nAll Network Params before pre-train")
#     network.print_params()
#     print("\nPre-train Layer 1")
#     recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
#     print("\nPre-train Layer 2")
#     recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=False)
#     print("\nAll Network Params after pre-train")
#     network.print_params()
#
#     # Fine-tune
#     print("\nFine-tune Network")
#     correct_prediction = tf.equal(tf.argmax(y, 1), y_)
#     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     print('   learning_rate: %f' % learning_rate)
#     print('   batch_size: %d' % batch_size)
#
#     for epoch in range(n_epoch):
#         start_time = time.time()
#         for X_train_a, y_train_a in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
#             feed_dict = {x: X_train_a, y_: y_train_a}
#             feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
#             feed_dict[set_keep['denoising1']] = 1    # disable denoising layer
#             sess.run(train_op, feed_dict=feed_dict)
#
#         if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#             print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#             train_loss, train_acc, n_batch = 0, 0, 0
#             for X_train_a, y_train_a in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
#                 dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
#                 feed_dict = {x: X_train_a, y_: y_train_a}
#                 feed_dict.update(dp_dict)
#                 err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#                 train_loss += err
#                 train_acc += ac
#                 n_batch += 1
#             print("   train loss: %f" % (train_loss/ n_batch))
#             print("   train acc: %f" % (train_acc/ n_batch))
#             val_loss, val_acc, n_batch = 0, 0, 0
#             for X_val_a, y_val_a in iterate_minibatches(X_val, y_val, batch_size, shuffle=True):
#                 dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
#                 feed_dict = {x: X_val_a, y_: y_val_a}
#                 feed_dict.update(dp_dict)
#                 err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#                 val_loss += err
#                 val_acc += ac
#                 n_batch += 1
#             print("   val loss: %f" % (val_loss/ n_batch))
#             print("   val acc: %f" % (val_acc/ n_batch))
#             try:
#                 visualize_W(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
#             except:
#                 raise Exception("# You should change visualize_W(), if you want to save the feature images for different dataset")
#
#     print('Evaluation')
#     test_loss, test_acc, n_batch = 0, 0, 0
#     for X_test_a, y_test_a in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
#         dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout layers
#         feed_dict = {x: X_test_a, y_: y_test_a}
#         feed_dict.update(dp_dict)
#         err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#         test_loss += err
#         test_acc += ac
#         n_batch += 1
#     print("   test loss: %f" % (test_loss/n_batch))
#     print("   test acc: %f" % (test_acc/n_batch))
#         # print("   test acc: %f" % np.mean(y_test == sess.run(y_op, feed_dict=feed_dict)))
#
#     # Add ops to save and restore all the variables.
#     # ref: https://www.tensorflow.org/versions/r0.8/how_tos/variables/index.html
#     saver = tf.train.Saver()
#     # you may want to save the model
#     save_path = saver.save(sess, "model.ckpt")
#     print("Model saved in file: %s" % save_path)
#     sess.close()
#
# def main_test_cnn_layer():
#     '''
#         Reimplementation of the tensorflow official MNIST CNN tutorials:
#         # https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html
#         # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py
#     '''
#     X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_dataset(shape=(-1, 28, 28, 1))
#
#     X_train = np.asarray(X_train, dtype=np.float32)
#     y_train = np.asarray(y_train, dtype=np.int64)
#     X_val = np.asarray(X_val, dtype=np.float32)
#     y_val = np.asarray(y_val, dtype=np.int64)
#     X_test = np.asarray(X_test, dtype=np.float32)
#     y_test = np.asarray(y_test, dtype=np.int64)
#
#     print('X_train.shape', X_train.shape)
#     print('y_train.shape', y_train.shape)
#     print('X_val.shape', X_val.shape)
#     print('y_val.shape', y_val.shape)
#     print('X_test.shape', X_test.shape)
#     print('y_test.shape', y_test.shape)
#     print('X %s   y %s' % (X_test.dtype, y_test.dtype))
#
#     sess = tf.InteractiveSession()
#
#     x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])   # [batch_size, height, width, channels]
#     y_ = tf.placeholder(tf.int64, shape=[None,])
#
#     network = InputLayer(x, name='input_layer')
#     network = Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
#                         strides=[1, 1, 1, 1],
#                         padding='SAME',
#                         name ='cnn_layer1')     # output: (?, 28, 28, 100)
#     network = Pool2dLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='pool_layer1',)   # output: (?, 14, 14, 100)
#     network = Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [5, 5, 32, 64], # 64 features for each 5x5 patch
#                         strides=[1, 1, 1, 1],
#                         padding='SAME',
#                         name ='cnn_layer2')     # output: (?, 14, 14, 32)
#     network = Pool2dLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='pool_layer2',)   # output: (?, 7, 7, 32)
#     network = FlattenLayer(network, name='flatten_layer')
#     network = DropoutLayer(network, keep=0.5, name='drop1')
#     network = DenseLayer(network, n_units=256, act = tf.nn.relu, name='relu1')
#     network = DropoutLayer(network, keep=0.5, name='drop2')
#     network = DenseLayer(network, n_units=10, act = identity, name='output_layer')
#
#     y = network.outputs
#
#     ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
#     cost = ce
#
#     correct_prediction = tf.equal(tf.argmax(y, 1), y_)
#     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#     # train
#     n_epoch = 500
#     batch_size = 128
#     learning_rate = 0.0001
#     print_freq = 10
#
#     train_params = network.all_params
#     train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)
#
#     sess.run(tf.initialize_all_variables())
#     network.print_params()
#     network.print_layers()
#
#     print('   learning_rate: %f' % learning_rate)
#     print('   batch_size: %d' % batch_size)
#
#     for epoch in range(n_epoch):
#         start_time = time.time()
#         for X_train_a, y_train_a in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
#             feed_dict = {x: X_train_a, y_: y_train_a}
#             feed_dict.update( network.all_drop )        # enable all dropout/dropconnect/denoising layers
#             sess.run(train_op, feed_dict=feed_dict)
#
#         if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
#             print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
#             train_loss, train_acc, n_batch = 0, 0, 0
#             for X_train_a, y_train_a in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
#                 dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
#                 feed_dict = {x: X_train_a, y_: y_train_a}
#                 feed_dict.update(dp_dict)
#                 err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#                 train_loss += err
#                 train_acc += ac
#                 n_batch += 1
#             print("   train loss: %f" % (train_loss/ n_batch))
#             print("   train acc: %f" % (train_acc/ n_batch))
#             val_loss, val_acc, n_batch = 0, 0, 0
#             for X_val_a, y_val_a in iterate_minibatches(X_val, y_val, batch_size, shuffle=True):
#                 dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout/dropconnect/denoising layers
#                 feed_dict = {x: X_val_a, y_: y_val_a}
#                 feed_dict.update(dp_dict)
#                 err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#                 val_loss += err
#                 val_acc += ac
#                 n_batch += 1
#             print("   val loss: %f" % (val_loss/ n_batch))
#             print("   val acc: %f" % (val_acc/ n_batch))
#             # try:
#             #     visualize_W(network.all_params[0].eval(), second=10, saveable=True, name='w1_'+str(epoch+1), fig_idx=2012)
#             # except:
#             #     raise Exception("# You should change visualize_W(), if you want to save the feature images for different dataset")
#
#     print('Evaluation')
#     test_loss, test_acc, n_batch = 0, 0, 0
#     for X_test_a, y_test_a in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
#         dp_dict = Layer.dict_to_one( network.all_drop )    # disable all dropout layers
#         feed_dict = {x: X_test_a, y_: y_test_a}
#         feed_dict.update(dp_dict)
#         err, ac = sess.run([cost, acc], feed_dict=feed_dict)
#         test_loss += err
#         test_acc += ac
#         n_batch += 1
#     print("   test loss: %f" % (test_loss/n_batch))
#     print("   test acc: %f" % (test_acc/n_batch))

def main_pg_pong2():
    '''
    Deep Reinforcement Learning: Pong from Pixels
    http://karpathy.github.io/2016/05/31/rl/
    Install OpenAI Gym
    https://github.com/openai/gym
    feedforward and backforward by using tensorflow
    '''
    import gym

    np.set_printoptions(threshold=np.nan)    # print all values for debug



    def prepro(I):
      """ pre-processing: 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
      # print(I.shape)    # (210, 160, 3)
      # visualize_frame(I)
      I = I[35:195] # crop , remove the scoreboard
      # visualize_frame(I)
      # print(I.shape)    # (160, 160, 3)
      I = I[::2,::2,0] # downsample by factor of 2
      # visualize_frame(I)
      # print(I.shape)    # (80, 80)

      I[I == 144] = 0 # erase background (background type 1)
      # visualize_frame(I)
      I[I == 109] = 0 # erase background (background type 2)
      # visualize_frame(I)
      I[I != 0] = 1 # everything else (paddles, ball) just set to 1
      # visualize_frame(I)
      return I.astype(np.float).ravel() # (6400,)

    def discount_rewards(r, gamma):
      """ take 1D float array of rewards and compute discounted reward """
      # print(r.shape)    # (1759, 1)
      discounted_r = np.zeros_like(r)
      running_add = 0
      # for t in reversed(xrange(0, r.size)):   # python2
      for t in reversed(range(0, r.size)):     # python3
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
      return discounted_r

    sess = tf.InteractiveSession()

    # hyperparameters
    H = 200 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = True # load model, resume from previous checkpoint?
    render = False # display the game screen
    # model initialization
    D = 80 * 80 # input dimensionality: 80x80 grid

    # build model
    x = tf.placeholder(tf.float32, shape=[None, D], name='x')               # observations
    # y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')              # network outputs
    rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')    # discount rewards
    labels_ = tf.placeholder(tf.float32, shape=[None, ], name='labels_')    # "fake label"

    network = InputLayer(x, name='input_layer')
    network = DenseLayer(network, n_units= H , act = tf.nn.relu, name='relu_layer')
    network = DenseLayer(network, n_units= 1 , act = tf.nn.sigmoid, name='output_layer')

    y = network.outputs

    # https://www.tensorflow.org/versions/r0.9/api_docs/python/train.html#optimizers
    cost = - tf.reduce_sum( tf.mul( tf.sub(labels_ , y), rewards) )    # NOT CORRECT ?
    # train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(cost)
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(cost)

    sess.run(tf.initialize_all_variables())

    if resume:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        saver.restore(sess, "model_pong.ckpt")

    network.print_params()

    # start game
    env = gym.make("Pong-v0")
    observation = env.reset()   # frame     (210, 160, 3)   60 Hz
    prev_x = None # used in computing the difference frame
    xs, drs, labels = [], [], []  # observations, discounted rewards and fake labels in a episode
    # xs, ys, drs = [], [], []    # observations, network outputs and rewards in a episode
    # xs, hs, dlogps, drs = [],[],[],[]   # X_train, hidden_out, grad, list_reward in a episode
    X = np.empty(shape=(0, D))            # observation in a batch
    # Y = np.empty(shape=(0, 1))           # p(x) policy network outputs
    L = np.empty(shape=(0, 1))             # fake labels
    R = np.empty(shape=(0, 1))             # f(x) reward function
    running_reward = None   # the averaged and discounted reward of recent episodes
    reward_sum = 0          # the sum of reward in a episode
    episode_number = 0      # episode index, each episode has a lot of games
    game_number = 0         # game index in a episode
    start_time = time.time()

    while True:
        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)   # 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        # compute the difference frame
        x_diff = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        feed_dict = {x: np.asarray([x_diff], dtype=np.float32)}   # (1, 6400)
        aprob = sess.run(y, feed_dict=feed_dict)    # [[ 0.5]]
        # ys.append(aprob)

        # choice action. 0: STOP  2: UP  3: DOWN    (aprob=1, action=2; aprob=0, action=3)
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

        # record various intermediates (needed later for backprop)
        xs.append(x_diff) # observation    all X_train in a episode
        label = 1 if action == 2 else 0 # a "fake label", (action == 2, fake_label = 1) (action == 3, fake_label = 0)
        labels.append(label)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
            # when one player win 21 time, done==True
            # print(observation.shape, reward, done, info)  # (210, 160, 3) 0.0 False {}
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1
            game_number = 0

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)   # xs -> numpy (1271, 6400) all X_train in a episode, change xs from list to np. n_action = 1271
            epr = np.vstack(drs)  # (1271, 1) list of rewards of a episode
            # epdlogp = np.vstack(dlogps)
            # epy = np.vstack(ys)
            elabels = np.vstack(labels)

            xs, drs, labels = [], [], []

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, gamma)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # store all info of a episode until the end of a batch
            X = np.vstack((X, epx))
            # Y = np.vstack((Y, epy))
            R = np.vstack((R, discounted_epr))
            L = np.vstack((L, elabels))

            # perform parameter update and save model every batch_size episodes
            if episode_number % batch_size == 0:
                print("Update model " + "!"*10)
                print(X.shape, L.ravel().shape, R.ravel().shape)
                feed_dict = {x: X, labels_: L.ravel(), rewards: R.ravel()}
                sess.run(train_op, feed_dict=feed_dict)

                X = np.empty(shape=(0, D))            # observation in a batch
                # Y = np.empty(shape=(0, 1))
                L = np.empty(shape=(0, 1))
                R = np.empty(shape=(0, 1))

                print("Save model " + "!"*10);
                saver = tf.train.Saver()
                save_path = saver.save(sess, "model_pong.ckpt")

                network.print_params()

            # recent performance
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None


        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print(('episode %d: game %d took %.5s s, reward: %f' % (episode_number, game_number, time.time()-start_time, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
            start_time = time.time()
            game_number += 1

# def main_pg_pong1():
#     ''' feedforward uses tensorflow '''
#     import gym
#
#     def visualize_frame(I, ion = True, second=5, fig_idx=12836):
#         ''' display a frame. Make sure OpenAI Gym render() is disable before using it. '''
#         if ion:
#             plt.ion()
#         fig = plt.figure(fig_idx)      # show all feature images
#
#         plt.imshow(I)
#         # plt.gca().xaxis.set_major_locator(plt.NullLocator())    # distable tick
#         # plt.gca().yaxis.set_major_locator(plt.NullLocator())
#
#         if ion:
#             plt.draw()
#             plt.pause(second)
#         else:
#             plt.show()
#
#     def sigmoid(x):
#       return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]
#
#     def prepro(I):
#       """ pre-processing: 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#       # print(I.shape)    # (210, 160, 3)
#       # visualize_frame(I)
#       I = I[35:195] # crop , remove the scoreboard
#       # visualize_frame(I)
#       # print(I.shape)    # (160, 160, 3)
#       I = I[::2,::2,0] # downsample by factor of 2
#       # visualize_frame(I)
#       # print(I.shape)    # (80, 80)
#
#       I[I == 144] = 0 # erase background (background type 1)
#       # visualize_frame(I)
#       I[I == 109] = 0 # erase background (background type 2)
#       # visualize_frame(I)
#       I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#       # visualize_frame(I)
#       return I.astype(np.float).ravel() # (6400,)
#
#     def discount_rewards(r, gamma):
#       """ take 1D float array of rewards and compute discounted reward """
#       # print(r.shape)    # (1759, 1)
#       discounted_r = np.zeros_like(r)
#       running_add = 0
#       # for t in reversed(xrange(0, r.size)):   # python2
#       for t in reversed(range(0, r.size)):     # python3
#         if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
#         running_add = running_add * gamma + r[t]
#         discounted_r[t] = running_add
#       return discounted_r
#
#     def policy_backward(eph, epdlogp, epx, model):
#       """ backward pass. (eph is array of intermediate hidden states) """
#       dW2 = np.dot(eph.T, epdlogp).ravel()
#       dh = np.outer(epdlogp, model['W2'])
#       dh[eph <= 0] = 0 # backpro prelu
#       dW1 = np.dot(dh.T, epx)
#       return {'W1':dW1, 'W2':dW2}
#
#     sess = tf.InteractiveSession()
#
#     # hyperparameters
#     H = 200 # number of hidden layer neurons
#     batch_size = 10 # every how many episodes to do a param update?
#     learning_rate = 1e-4
#     gamma = 0.99 # discount factor for reward
#     decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
#     resume = True # load model, resume from previous checkpoint?
#     render = True # display the game screen
#
#     # model initialization
#     D = 80 * 80 # input dimensionality: 80x80 grid
#
#
#     # build model
#     x = tf.placeholder(tf.float32, shape=[None, D], name='x')               # observations
#     # y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')              # network outputs
#     rewards = tf.placeholder(tf.float32, shape=[None, ], name='rewards')    # discount rewards
#     labels_ = tf.placeholder(tf.float32, shape=[None, ], name='labels_')    # "fake label"
#
#     network = InputLayer(x, name='input_layer')
#     network = DenseLayer(network, n_units= H , act = tf.nn.relu, name='relu_layer')
#     network = DenseLayer(network, n_units= 1 , act = tf.nn.sigmoid, name='output_layer')
#
#     y = network.outputs
#
#     sess.run(tf.initialize_all_variables())
#
#     if resume:
#         print("Load existing model " + "!"*10)
#         saver = tf.train.Saver()
#         saver.restore(sess, "model_pong.ckpt")
#
#     network.print_params()
#
#     # initialize gradients buffer
#     grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch         # python3
#     rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory                                          # python3
#
#     env = gym.make("Pong-v0")
#     observation = env.reset()   # frame
#
#     prev_x = None # used in computing the difference frame
#     xs, hs, dlogps, drs = [],[],[],[]   # X_train, hidden_out, grad, list_reward in a episode
#     running_reward = None   # the averaged and discounted reward of recent episodes
#     reward_sum = 0          # the sum of reward in a episode
#     episode_number = 0      # episode index, each episode has a lot of games
#     game_number = 0         # game index in a episode
#     start_time = time.time()
#     while True:
#         if render: env.render()
#
#         # preprocess the observation, set input to network to be difference image
#         cur_x = prepro(observation)   # 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
#         # compute the difference frame
#         x = cur_x - prev_x if prev_x is not None else np.zeros(D)
#         prev_x = cur_x
#
#         # forward the policy network and sample an action from the returned probability
#         # aprob: network output;  h: hidden layer outputs
#         aprob, h = policy_forward(x, model)
#             #   print(aprob, h.shape)   # 0.5  (200,)
#         # choice action. 0: STOP  2: UP  3: DOWN
#         action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
#             #   print(np.random.uniform())    # 0 ~ 1 random value
#
#         # record various intermediates (needed later for backprop)
#         xs.append(x) # observation    all X_train in a episode
#         hs.append(h) # hidden state   all hidden outputs in a episode
#         y = 1 if action == 2 else 0 # a "fake label", (action == 2, y = 1) (action == 3, y = 0)
#         dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
#
#         # step the environment and get new measurements
#         #   when one player win 21 time, done==True
#         observation, reward, done, info = env.step(action)
#             # print(observation.shape, reward, done, info)  # (210, 160, 3) 0.0 False {}
#         reward_sum += reward
#
#         drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
#
#         if done: # an episode finished
#             episode_number += 1
#             game_number = 0
#
#             # stack together all inputs, hidden states, action gradients, and rewards for this episode
#             epx = np.vstack(xs)   # (1271, 6400) all X_train in a batch, change xs from list to np. n_action = 1271
#             eph = np.vstack(hs)   # (1271, 200)
#
#             epdlogp = np.vstack(dlogps) # (1271, 1) grad for update
#             epr = np.vstack(drs)        # (1271, 1) list of rewards
#             # print(epx.shape)
#             # print(eph.shape)
#             # print(epdlogp.shape)
#             # print(epr.shape)
#             # exit()
#
#             xs, hs, dlogps, drs = [],[],[],[] # reset array memory for next episode
#
#             # compute the discounted reward backwards through time
#             discounted_epr = discount_rewards(epr, gamma)
#             # standardize the rewards to be unit normal (helps control the gradient estimator variance)
#             discounted_epr -= np.mean(discounted_epr)
#             discounted_epr /= np.std(discounted_epr)
#
#             epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
#             grad = policy_backward(eph, epdlogp, epx, model)
#             for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
#
#             # perform rmsprop parameter update every batch_size episodes
#             if episode_number % batch_size == 0:
#                 print("Update model")
#                 # for k, v in model.items():
#                 #     g = grad_buffer[k] # gradient
#                 #     rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
#                 #     model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
#                 #     grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
#                 for k, v in enumerate(network.all_params()):
#                     g = grad_buffer[k] # gradient
#                     rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
#                     model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
#                     grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
#
#                 network.print_params()
#
#             # recent performance
#             running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
#             print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
#             reward_sum = 0
#             observation = env.reset() # reset env
#             prev_x = None
#
#             # save model
#             # if episode_number % 30 == 0: print('Save model'); pickle.dump(model, open('model_pong.p', 'wb'))
#             if episode_number % 30 == 0:
#                 print("Save model " + "!"*10);
#                 saver = tf.train.Saver()
#                 save_path = saver.save(sess, "model_pong.ckpt")
#
#         if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
#             print(('episode %d: game %d took %.5s s, reward: %f' % (episode_number, game_number, time.time()-start_time, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
#             start_time = time.time()
#             game_number += 1



if __name__ == '__main__':
    sess = set_gpu_fraction(gpu_fraction = 0.3)
    try:
        # main_test_layers(model='relu')        # model = relu, resnet, dropconnect
        # main_test_denoise_AE(model='relu')    # model = relu, sigmoid
        # main_test_stacked_denoise_AE(model='relu')        # model = relu, sigmoid
        # main_test_cnn_layer()
        # main_pg_pong1()                       # To Do
        main_pg_pong2()                        # To Do
        exit_tf(sess)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        exit_tf(sess)
