#! /usr/bin/python
# -*- coding: utf-8 -*-

import six
import time
from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer import files
from tensorlayer import iterate
from tensorlayer import utils
from tensorlayer import visualize

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import protected_method
from tensorlayer.decorators import private_method

__all__ = [
    'LayersConfig',
    'TF_GRAPHKEYS_VARIABLES',
    'flatten_reshape',
    'clear_layers_name',
    'set_name_reuse',
    'initialize_rnn_state',
    'print_all_variables',
    'get_variables_with_name',
    'get_layers_with_name',
    'list_remove_repeat',
    'merge_networks',
    'initialize_global_variables',
    'Layer',
    'DenseLayer',
    'ReconLayer',
    'DropoutLayer',
    'GaussianNoiseLayer',
    'DropconnectDenseLayer',
]


@six.add_metaclass(ABCMeta)
class LayersConfig(object):

    tf_dtype = tf.float32  # TensorFlow DType
    set_keep = {}  # A dictionary for holding tf.placeholders

    @abstractmethod
    def __init__(self):
        pass


TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


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


@deprecated("2018-06-30", "TensorLayer relies on TensorFlow to check naming.")
def clear_layers_name():
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')


@deprecated("2018-06-30", "TensorLayer relies on TensorFlow to check name reusing.")
def set_name_reuse(enable=True):
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')


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
        t_vars = tf.global_variables()
        logging.info("  [*] printing global variables")

    for idx, v in enumerate(t_vars):
        logging.info("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))


@deprecated_alias(printable='verbose', end_support_version=1.9)  # TODO remove this line for the 1.9 release
def get_variables_with_name(name=None, train_only=True, verbose=False):
    """Get a list of TensorFlow variables by a given name scope.

    Parameters
    ----------
    name : str
        Get the variables that contain this name.
    train_only : boolean
        If Ture, only get the trainable variables.
    verbose : boolean
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
        t_vars = tf.global_variables()

    d_vars = [var for var in t_vars if name in var.name]

    if verbose:
        for idx, v in enumerate(d_vars):
            logging.info("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))

    return d_vars


@deprecated_alias(printable='verbose', end_support_version=1.9)  # TODO remove this line for the 1.9 release
def get_layers_with_name(net, name="", verbose=False):
    """Get a list of layers' output in a network by a given name scope.

    Parameters
    -----------
    net : :class:`Layer`
        The last layer of the network.
    name : str
        Get the layers' output that contain this name.
    verbose : boolean
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

            if verbose:
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
    if sess is None:
        raise AssertionError('The session must be defined')

    sess.run(tf.global_variables_initializer())


class Layer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    prev_layer : :class:`Layer` or None
        Previous layer (optional), for adding all properties of previous layer(s) to this layer.
    act : activation function (None by default)
        The activation function of this layer.
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

    Examples
    ---------

    - Define model

    >>> x = tf.placeholder("float32", [None, 100])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.DenseLayer(n, 80, name='d1')
    >>> n = tl.layers.DenseLayer(n, 80, name='d2')

    - Get information

    >>> print(n)
    ... Last layer is: DenseLayer (d2) [None, 80]
    >>> n.print_layers()
    ... [TL]   layer   0: d1/Identity:0        (?, 80)            float32
    ... [TL]   layer   1: d2/Identity:0        (?, 80)            float32
    >>> n.print_params(False)
    ... [TL]   param   0: d1/W:0               (100, 80)          float32_ref
    ... [TL]   param   1: d1/b:0               (80,)              float32_ref
    ... [TL]   param   2: d2/W:0               (80, 80)           float32_ref
    ... [TL]   param   3: d2/b:0               (80,)              float32_ref
    ... [TL]   num of params: 14560
    >>> n.count_params()
    ... 14560

    - Slicing the outputs

    >>> n2 = n[:, :30]
    >>> print(n2)
    ... Last layer is: Layer (d2) [None, 30]

    - Iterating the outputs

    >>> for l in n:
    >>>    print(l)
    ... Tensor("d1/Identity:0", shape=(?, 80), dtype=float32)
    ... Tensor("d2/Identity:0", shape=(?, 80), dtype=float32)

    """
    # Added to allow auto-completion

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, act=None, name=None, *args, **kwargs):

        self.inputs = None
        self.outputs = None

        self.all_layers = list()
        self.all_params = list()
        self.all_drop = dict()

        if name is None:
            raise ValueError('Layer must have a name.')

        for key in kwargs.keys():
            setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        self.act = act if act not in [None, tf.identity] else None

        scope_name = tf.get_variable_scope().name

        self.name = scope_name + '/' + name if scope_name else name

        if isinstance(prev_layer, Layer):
            # 1. for normal layer have only 1 input i.e. DenseLayer
            # Hint : list(), dict() is pass by value (shallow), without them,
            # it is pass by reference.

            self.inputs = prev_layer.outputs

            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)

        elif isinstance(prev_layer, list):
            # 2. for layer have multiply inputs i.e. ConcatLayer

            self.inputs = [layer.outputs for layer in prev_layer]

            self._add_layers(sum([l.all_layers for l in prev_layer], []))
            self._add_params(sum([l.all_params for l in prev_layer], []))
            self._add_dropout_layers(sum([list(l.all_drop.items()) for l in prev_layer], []))

        elif isinstance(prev_layer, tf.Tensor) or isinstance(prev_layer, tf.Variable):  # placeholders
            if self.__class__.__name__ not in ['InputLayer', 'OneHotInputLayer', 'Word2vecEmbeddingInputlayer',
                                               'EmbeddingInputlayer', 'AverageEmbeddingInputlayer']:

                _err = "Please use `tl.layers.InputLayer` to convert Tensor/Placeholder to a TL layer"
                logging.error(_err)
                raise RuntimeError(_err)

            self.inputs = prev_layer

        elif prev_layer is not None:
            # 4. tl.models
            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)

            if hasattr(prev_layer, "outputs"):
                self.inputs = prev_layer.outputs

    def print_params(self, details=True, session=None):
        """Print all info of parameters in the network"""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    # logging.info("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                    val = p.eval(session=session)
                    logging.info(
                        "  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                            i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()
                        )
                    )
                except Exception as e:
                    logging.info(str(e))
                    raise Exception(
                        "Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False)."
                    )
            else:
                logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network"""

        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info(
                "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            )

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

        net_new = Layer(prev_layer=None, name=self.name)

        net_new.inputs = self.inputs
        net_new.outputs = self.outputs[key]

        net_new._add_layers(self.all_layers[:-1])
        net_new._add_layers(net_new.outputs)

        net_new._add_params(self.all_params)

        net_new._add_dropout_layers(self.all_drop)

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

    @protected_method
    def _add_layers(self, layers):
        if isinstance(layers, list):
            try:  # list of class Layer
                new_layers = [layer.outputs for layer in layers]
                self.all_layers.extend(list(new_layers))

            except AttributeError:  # list of tf.Tensor
                self.all_layers.extend(list(layers))

        else:
            self.all_layers.append(layers)

        self.all_layers = list_remove_repeat(self.all_layers)

    @protected_method
    def _add_params(self, params):

        if isinstance(params, list):
            self.all_params.extend(list(params))

        else:
            self.all_params.append(params)

        self.all_params = list_remove_repeat(self.all_params)

    @protected_method
    def _add_dropout_layers(self, drop_layers):
        if isinstance(drop_layers, dict) or isinstance(drop_layers, list):
            self.all_drop.update(dict(drop_layers))

        elif isinstance(drop_layers, tuple):
            self.all_drop.update(list(drop_layers))

        else:
            raise ValueError()

    @private_method
    def _apply_activation(self, logits):
        return self.act(logits) if self.act is not None else logits

    @private_method
    def _argument_dict_checkup(self, args):

        if not isinstance(args, dict) and args is not None:
            err_msg = "One of the argument given to %s should be formatted as a dictionnary" % self.__class__.__name__
            logging.error(err_msg)
            raise AssertionError(err_msg)

        return args if args is not None else {}


class DenseLayer(Layer):
    """The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    prev_layer : :class:`Layer`
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dense',
    ):

        super(DenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DenseLayer  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else '- No Activation')
        )

        self.n_units = n_units

        if self.inputs.get_shape().ndims != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )

            self.outputs = tf.matmul(self.inputs, W)

            if b_init is not None:
                try:
                    b = tf.get_variable(
                        name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        if b_init is not None:
            self._add_params([W, b])
        else:
            self._add_params(W)


class ReconLayer(DenseLayer):
    """A reconstruction layer for :class:`DenseLayer` to implement AutoEncoder.

    It is often used to pre-train the previous :class:`DenseLayer`

    Parameters
    ----------
    prev_layer : :class:`Layer`
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
    >>> x = tf.placeholder(tf.float32, shape=(None, 784))
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, n_units=196, act=tf.nn.sigmoid, name='dense')
    >>> recon = tl.layers.ReconLayer(net, x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon')
    >>> sess = tf.InteractiveSession()
    >>> tl.layers.initialize_global_variables(sess)
    >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    >>> recon.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name=None, n_epoch=500, batch_size=128, print_freq=1, save=True, save_name='w1pre_')

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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            x_recon=None,
            n_units=784,
            act=tf.nn.softplus,
            name='recon',
    ):
        super(ReconLayer, self).__init__(prev_layer=prev_layer, n_units=n_units, act=act, name=name)

        logging.info("ReconLayer %s" % self.name)

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
        L2_w = tf.contrib.layers.l2_regularizer(lambda_l2_w)(
            self.train_params[0]
        ) + tf.contrib.layers.l2_regularizer(lambda_l2_w)(self.train_params[2])  # faster than the code below
        # L2_w = lambda_l2_w * tf.reduce_mean(tf.square(self.train_params[0])) + lambda_l2_w * tf.reduce_mean( tf.square(self.train_params[2]))

        # DropNeuro
        # P_o = cost.lo_regularizer(0.03)(
        #     self.train_params[0])  # + cost.lo_regularizer(0.5)(self.train_params[2])    # <haodong>: if add lo on decoder, no neuron will be broken
        # P_i = cost.li_regularizer(0.03)(self.train_params[0])  # + cost.li_regularizer(0.001)(self.train_params[2])

        # L1 of activation outputs
        activation_out = self.all_layers[-2]
        L1_a = 0.001 * tf.reduce_mean(
            activation_out
        )  # <haodong>:  theano: T.mean( self.a[i] )         # some neuron are broken, white and black
        # L1_a = 0.001 * tf.reduce_mean( tf.reduce_sum(activation_out, 0) )         # <haodong>: some neuron are broken, white and black
        # L1_a = 0.001 * 100 * tf.reduce_mean( tf.reduce_sum(activation_out, 1) )   # <haodong>: some neuron are broken, white and black
        # KL Divergence
        beta = 4
        rho = 0.15
        p_hat = tf.reduce_mean(activation_out, 0)  # theano: p_hat = T.mean( self.a[i], axis=0 )

        KLD = beta * tf.reduce_sum(
            rho * tf.log(tf.divide(rho, p_hat)) + (1 - rho) * tf.log((1 - rho) / (tf.subtract(float(1), p_hat)))
        )

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

        self.train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False).minimize(self.cost, var_list=self.train_params)
        # self.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(self.cost, var_list=self.train_params)

    def pretrain(
            self, sess, x, X_train, X_val, denoise_name=None, n_epoch=100, batch_size=128, print_freq=10, save=True,
            save_name='w1pre_'
    ):
        # ====================================================
        #
        # You need to modify the cost function in __init__() so as to
        # get your own pre-train method.
        #
        # ====================================================
        logging.info("     [*] %s start pretrain" % self.name)
        logging.info("     batch_size: %d" % batch_size)
        if denoise_name:
            logging.info("     denoising layer keep: %f" % self.all_drop[LayersConfig.set_keep[denoise_name]])
            dp_denoise = self.all_drop[LayersConfig.set_keep[denoise_name]]
        else:
            logging.info("     no denoising layer")

        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, _ in iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                dp_dict = utils.dict_to_one(self.all_drop)
                if denoise_name:
                    dp_dict[LayersConfig.set_keep[denoise_name]] = dp_denoise
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
                            self.train_params[0].eval(), second=10, saveable=True, shape=[28, 28],
                            name=save_name + str(epoch + 1), fig_idx=2012
                        )
                        files.save_npz([self.all_params[0]], name=save_name + str(epoch + 1) + '.npz')
                    except Exception:
                        raise Exception(
                            "You should change the visualize.W() in ReconLayer.pretrain(), if you want to save the feature images for different dataset"
                        )


class DropoutLayer(Layer):
    """
    The :class:`DropoutLayer` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    prev_layer : :class:`Layer`
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            keep=0.5,
            is_fix=False,
            is_train=True,
            seed=None,
            name='dropout_layer',
    ):
        super(DropoutLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("DropoutLayer %s: keep:%f is_fix:%s" % (self.name, keep, is_fix))

        if is_train is False:
            logging.info("  skip DropoutLayer")
            self.outputs = prev_layer.outputs

        else:

            # The name of placeholder for keep_prob is the same with the name of the Layer.
            if is_fix:
                self.outputs = tf.nn.dropout(self.inputs, keep, seed=seed, name=name)
            else:
                LayersConfig.set_keep[name] = tf.placeholder(LayersConfig.tf_dtype)
                self.outputs = tf.nn.dropout(self.inputs, LayersConfig.set_keep[name], seed=seed, name=name)  # 1.2

            if is_fix is False:
                self.all_drop.update({LayersConfig.set_keep[name]: keep})

            self._add_layers(self.outputs)


class GaussianNoiseLayer(Layer):
    """
    The :class:`GaussianNoiseLayer` class is noise layer that adding noise with
    gaussian distribution to the activation.

    Parameters
    ------------
    prev_layer : :class:`Layer`
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

    Examples
    ----------
    >>> x = tf.placeholder(tf.float32, shape=(100, 784))
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, name='dense3')
    >>> net = tl.layers.GaussianNoiseLayer(net, name='gaussian')
    ... (64, 100)

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            mean=0.0,
            stddev=1.0,
            is_train=True,
            seed=None,
            name='gaussian_noise_layer',
    ):
        super(GaussianNoiseLayer, self).__init__(prev_layer=prev_layer, name=name)

        if is_train is False:
            logging.info("  skip GaussianNoiseLayer")
            self.outputs = prev_layer.outputs

        else:
            logging.info("GaussianNoiseLayer %s: mean:%f stddev:%f" % (self.name, mean, stddev))
            with tf.variable_scope(name):
                # noise = np.random.normal(0.0 , sigma , tf.to_int64(self.inputs).get_shape())
                noise = tf.random_normal(shape=self.inputs.get_shape(), mean=mean, stddev=stddev, seed=seed)
                self.outputs = self.inputs + noise

            self._add_layers(self.outputs)


class DropconnectDenseLayer(Layer):
    """
    The :class:`DropconnectDenseLayer` class is :class:`DenseLayer` with DropConnect
    behaviour which randomly removes connections between this layer and the previous
    layer according to a keeping probability.

    Parameters
    ----------
    prev_layer : :class:`Layer`
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            keep=0.5,
            n_units=100,
            act=None,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='dropconnect_layer',
    ):
        super(DropconnectDenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)

        logging.info(
            "DropconnectDenseLayer %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else '- No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )
            b = tf.get_variable(
                name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
            )
            # self.outputs = tf.matmul(self.inputs, W) + b

            LayersConfig.set_keep[name] = tf.placeholder(tf.float32)

            W_dropcon = tf.nn.dropout(W, LayersConfig.set_keep[name])

            self.outputs = self._apply_activation(tf.matmul(self.inputs, W_dropcon) + b)

        self.all_drop.update({LayersConfig.set_keep[name]: keep})
        self._add_layers(self.outputs)
        self._add_params([W, b])
