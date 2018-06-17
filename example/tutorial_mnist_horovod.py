#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def make_dataset(images, labels):
    ds1 = tf.data.Dataset.from_tensor_slices(images)
    ds2 = tf.data.Dataset.from_tensor_slices(np.array(labels, dtype=np.int64))
    return tf.data.Dataset.zip((ds1, ds2))


def make_network(x, y_):
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1', is_fix=True)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2', is_fix=True)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3', is_fix=True)
        network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
        return network, tl.cost.cross_entropy(network.outputs, y_, name='cost')



# load mnist data
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
dataset = make_dataset(X_train, y_train)

trainer = tl.distributed.SimpleTrainer(
    network_and_cost_func=make_network, dataset=dataset, optimizer=tf.train.RMSPropOptimizer, optimizer_args={
        'learning_rate': 0.001}
)
trainer.train_to_end()
