#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'get_collection_trainable',
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
]


def get_collection_trainable(name=''):
    variables = []
    for p in tf.trainable_variables():
        # print(p.name.rpartition('/')[0], self.name)
        if p.name.rpartition('/')[0] == name:
            variables.append(p)
    return variables


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
    >>> ... [None, 62, 62, 32]
    >>> network = tl.layers.flatten_reshape(network)
    >>> print(network.get_shape()[:].as_list()[1:])
    >>> ... [None, 123008]
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
        return c, h
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


@deprecated("2018-09-30", "This API is deprecated in favor of `tf.global_variables_initializer`.")
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
