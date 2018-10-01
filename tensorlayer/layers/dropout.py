#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'DropoutLayer',
]


class DropoutLayer(Layer):
    """
    The :class:`DropoutLayer` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    is_fix : boolean
        Fixing probability or nor. Default is False.
        If True, the keeping probability is fixed and cannot be changed via `feed_dict`.
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

    def __init__(
        self,
        keep=0.5,
        is_fix=False,
        seed=None,
        name='dropout_layer',
    ):
        self.keep = keep
        self.is_fix = is_fix
        self.seed = seed
        self.name = name

        super(DropoutLayer, self).__init__()

    def __str__(self):

        additional_str = []

        if self._temp_data['is_train']:

            try:
                additional_str.append("keep: %f" % self.keep)
            except AttributeError:
                pass

            try:
                additional_str.append("is_fix: %s" % self.is_fix)
            except AttributeError:
                pass

            return self._str(additional_str)

        else:
            return self._skipped_layer_str()

    def build(self):

        if self._temp_data['is_train']:

            with tf.variable_scope(self.name):
                # The name of placeholder for keep_prob is the same with the name of the Layer.
                if self.is_fix:
                    self._temp_data['outputs'] = tf.nn.dropout(
                        self._temp_data['inputs'], self.keep, seed=self.seed, name="dropout_op"
                    )

                else:
                    keep_plh = tf.placeholder(self._temp_data['inputs'].dtype, shape=())

                    self._add_local_drop_plh(keep_plh, self.keep)

                    LayersConfig.set_keep[self.name] = keep_plh

                    self._temp_data['outputs'] = tf.nn.dropout(
                        self._temp_data['inputs'], keep_plh, seed=self.seed, name="dropout_op"
                    )

        else:
            self._temp_data['outputs'] = self._temp_data['inputs']
