#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated
from tensorlayer.decorators import deprecated_alias

__all__ = [
    'cabs',
    'clear_layers_name',
    'compute_alpha',
    'flatten_reshape',
    'get_collection_trainable',
    'get_layers_with_name',
    'get_variables_with_name',
    'initialize_global_variables',
    'initialize_rnn_state',
    'list_remove_repeat',
    'merge_networks',
    'print_all_variables',
    'quantize',
    'quantize_active',
    'quantize_weight',
    'set_name_reuse',
    'ternary_operation',
]

########## Module Public Functions ##########


def cabs(x):
    return tf.minimum(1.0, tf.abs(x), name='cabs')


@deprecated(date="2018-06-30", instructions="TensorLayer relies on TensorFlow to check naming")
def clear_layers_name():
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')


def compute_alpha(x):
    """
    Computing the scale parameter.
    """
    threshold = _compute_threshold(x)
    alpha1_temp1 = tf.where(tf.greater(x, threshold), x, tf.zeros_like(x, tf.float32))
    alpha1_temp2 = tf.where(tf.less(x, -threshold), x, tf.zeros_like(x, tf.float32))
    alpha_array = tf.add(alpha1_temp1, alpha1_temp2, name=None)
    alpha_array_abs = tf.abs(alpha_array)
    alpha_array_abs1 = tf.where(
        tf.greater(alpha_array_abs, 0), tf.ones_like(alpha_array_abs, tf.float32),
        tf.zeros_like(alpha_array_abs, tf.float32)
    )
    alpha_sum = tf.reduce_sum(alpha_array_abs)
    n = tf.reduce_sum(alpha_array_abs1)
    alpha = tf.div(alpha_sum, n)
    return alpha


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
    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    >>> # Convolution Layer with 32 filters and a kernel size of 5
    >>> network = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    >>> # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
    >>> network = tf.layers.max_pooling2d(network, 2, 2)
    >>> print(network.get_shape()[:].as_list())
    >>> [None, 62, 62, 32]
    >>> network = tl.layers.flatten_reshape(network)
    >>> print(network.get_shape()[:].as_list()[1:])
    >>> [None, 123008]
    """

    dim = 1
    for d in variable.get_shape()[1:].as_list():
        dim *= d
    return tf.reshape(variable, shape=[-1, dim], name=name)


def get_collection_trainable(name=''):
    variables = []
    for p in tf.trainable_variables():
        # print(p.name.rpartition('/')[0], self.name)
        if p.name.rpartition('/')[0] == name:
            variables.append(p)
    return variables


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
    >>> import tensorlayer as tl
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
    >>> import tensorlayer as tl
    >>> dense_vars = tl.layers.get_variables_with_name('dense', True, True)

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


@deprecated(date="2018-09-30", instructions="This API is deprecated in favor of `tf.global_variables_initializer`")
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
        return c, h
    else:
        new_state = state.eval(feed_dict=feed_dict)
        return new_state


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
    [2, 3, 4]

    """
    y = []
    for i in x:
        if i not in y:
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
    >>> import tensorlayer as tl
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


def quantize(x):
    # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
    with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
        return tf.sign(x)


def quantize_active(x, bitA):
    if bitA == 32:
        return x
    return _quantize_dorefa(x, bitA)


def quantize_weight(x, bitW, force_quantization=False):
    G = tf.get_default_graph()
    if bitW == 32 and not force_quantization:
        return x
    if bitW == 1:  # BWN
        with G.gradient_override_map({"Sign": "Identity"}):
            E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
            return tf.sign(x / E) * E
    x = tf.clip_by_value(x * 0.5 + 0.5, 0.0, 1.0)  # it seems as though most weights are within -1 to 1 region anyways
    return 2 * _quantize_dorefa(x, bitW) - 1


@deprecated(date="2018-06-30", instructions="TensorLayer relies on TensorFlow to check name reusing")
def set_name_reuse(enable=True):
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')


def ternary_operation(x):
    """
    Ternary operation use threshold computed with weights.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        threshold = _compute_threshold(x)
        x = tf.sign(tf.add(tf.sign(tf.add(x, threshold)), tf.sign(tf.add(x, -threshold))))
        return x


########## Module Private Functions ##########


@tf.RegisterGradient("TL_Sign_QuantizeGrad")
def _quantize_grad(op, grad):
    """Clip and binarize tensor using the straight through estimator (STE) for the gradient. """
    return tf.clip_by_value(grad, -1, 1)


def _quantize_dorefa(x, k):
    G = tf.get_default_graph()
    n = float(2**k - 1)
    with G.gradient_override_map({"Round": "Identity"}):
        return tf.round(x * n) / n


def _compute_threshold(x):
    """
    ref: https://github.com/XJTUWYD/TWN
    Computing the threshold.
    """
    x_sum = tf.reduce_sum(tf.abs(x), reduction_indices=None, keepdims=False, name=None)
    threshold = tf.div(x_sum, tf.cast(tf.size(x), tf.float32), name=None)
    threshold = tf.multiply(0.7, threshold, name=None)
    return threshold
