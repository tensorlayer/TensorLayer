#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Dense',
]


class Dense(Layer):
    # FIXME: documentation update needed
    """The :class:`Dense` class is a fully connected layer.

    Parameters
    ----------
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
    name : None or str
        A unique layer name.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input(x, name='input')
    >>> net = tl.layers.Dense(net, 800, act=tf.nn.relu, name='relu')

    Without native TensorLayer APIs, you can do as follow.

    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
            self,
            n_units=100,
            act=None,
            # TODO: how to support more initializers
            # W_init=tf.truncated_normal_initializer(stddev=0.1),
            # b_init=tf.constant_initializer(value=0.0),
            W_init=tf.initializers.truncated_normal,
            b_init=tf.initializers.constant,
            W_init_args=None,
            b_init_args=None,
            name=None,  # 'dense',
    ):

        # super(Dense, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)

        self.n_units = n_units
        self.act = act
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args

        # self.n_in = int(self.inputs.get_shape()[-1])
        # self.inputs_shape = self.inputs.shape.as_list() #
        # self.outputs_shape = [self.inputs_shape[0], n_units]

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    '''
    def build(self, inputs):
        self.W = tf.get_variable(
            name='W', shape=(self.n_in, self.n_units), initializer=self.W_init, dtype=LayersConfig.tf_dtype,
            **self.W_init_args
        )
        if self.b_init is not None:
            try:
                self.b = tf.get_variable(
                    name='b', shape=(self.n_units), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
                    **self.b_init_args
                )
            except Exception:  # If initializer is a constant, do not specify shape.
                self.b = tf.get_variable(
                    name='b', initializer=self.b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                )
        self.get_weights(self.W, self.b)
    '''

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
        shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init, init_args=self.W_init_args)
        self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init, init_args=self.b_init_args)
        # outputs_shape = [inputs_shape[0], self.n_units]
        # return outputs_shape

    '''
    def forward(self, inputs, is_train):
        outputs = tf.matmul(inputs, self.W)
        if self.b_init is not None:
            outputs = tf.add(z, self.b)
        outputs = self.act(outputs)
        return outputs
    '''

    def forward(self, inputs, is_train):
        y = tf.matmul(inputs, self.W)
        z = tf.add(y, self.b)
        if self.act:
            z = self.act(z)
        return z


if __name__ == "__main__":
    # test

    from tensorlayer.layers import Input
    from tensorlayer.models import Model

    def eager_test():
        tf.enable_eager_execution()

        def generator(inputs_shape):
            innet = Input(inputs_shape)
            net = Dense(n_units=64, act=tf.nn.relu)(innet)
            net = Dense(n_units=64, act=tf.nn.relu)(net)
            net1 = Dense(n_units=1, act=tf.nn.relu)(net)
            net2 = Dense(n_units=5, act=tf.nn.relu)(net)

            G = Model(inputs=innet, outputs=[net1, net2])
            return G, net2

        latent_space_size = 100
        G, net2 = generator((None, latent_space_size))
        inputs = np.zeros([100, 100], dtype="float32")
        inputs = tf.convert_to_tensor(inputs)
        outputs_train = G(inputs, True)
        outputs_test = G(inputs, False)
        print(outputs_train, [_.shape for _ in outputs_train])
        print(outputs_test, [_.shape for _ in outputs_test])

    def graph_test():

        def disciminator(inputs_shape):
            innet = Input(inputs_shape)
            net = Dense(n_units=32, act=tf.nn.relu)(innet)
            net1 = Dense(n_units=1, act=tf.nn.relu)(net)
            net2 = Dense(n_units=5, act=tf.nn.relu)(net)
            D = Model(inputs=innet, outputs=[net1, net2])
            return D

        inputs = tf.placeholder(shape=[None, 100], dtype=tf.float32)
        D = disciminator(inputs_shape=[None, 100])
        outputs_train = D(inputs, is_train=True)
        outputs_test = D(inputs, is_train=False)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        real_inputs = np.ones((100, 100))
        real_outputs_train = sess.run(outputs_train, feed_dict={inputs: real_inputs})
        real_outputs_test = sess.run(outputs_test, feed_dict={inputs: real_inputs})
        print(real_outputs_train, [_.shape for _ in real_outputs_train])
        print(real_outputs_test, [_.shape for _ in real_outputs_test])

    # eager_test()
    graph_test()
