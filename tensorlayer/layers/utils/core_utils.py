#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated

from tensorlayer import logging

__all__ = [
    'get_collection_trainable',
    'get_layers_with_name',
    'get_variables_with_name',
    'print_all_variables',
    'clear_layers_name',
    'initialize_global_variables',
    'set_name_reuse',
]


@deprecated(
    end_support_version="2.0.0",
    instructions="This function will be removed in TL 2.0.0 in favor of `tf.get_collection`"
)
def get_collection_trainable(name=''):
    variables = []
    for p in tf.trainable_variables():
        # print(p.name.rpartition('/')[0], self.name)
        if p.name.rpartition('/')[0] == name:
            variables.append(p)
    return variables


@deprecated(
    end_support_version="2.0.0",
    instructions="This function will be removed in TL 2.0.0 in favor of `tf.get_collection`"
)
@deprecated_alias(printable='verbose', end_support_version="2.0.0")  # TODO: remove this line before releasing TL 2.0.0
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


@deprecated_alias(printable='verbose', end_support_version="2.0.0")  # TODO: remove this line before releasing TL 2.0.0
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


@deprecated(end_support_version="2.0.0", instructions="TensorLayer relies on TensorFlow to check naming")
def clear_layers_name():
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')


@deprecated(
    end_support_version="2.0.0", instructions="This API is deprecated in favor of `tf.global_variables_initializer`"
)
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


@deprecated(end_support_version="2.0.0", instructions="TensorLayer relies on TensorFlow to check name reusing")
def set_name_reuse(enable=True):
    logging.warning('this method is DEPRECATED and has no effect, please remove it from your code.')
