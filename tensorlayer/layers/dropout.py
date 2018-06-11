#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'DropoutLayer',
]


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

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
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

        logging.info("DropoutLayer %s: keep: %f is_fix: %s" % (self.name, keep, is_fix))

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
