#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def make_dataset(images, labels):
    ds1 = tf.data.Dataset.from_tensor_slices(images)
    ds2 = tf.data.Dataset.from_tensor_slices(np.array(labels, dtype=np.int64))
    return tf.data.Dataset.zip((ds1, ds2))


def model(x, y_, is_train):
    # Add droptout layers to the network during training
    with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
        network = tl.layers.InputLayer(x, name='input')
        network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
        network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3', is_fix=True, is_train=is_train)
        network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
    return network


def build_train(x, y_):
    net = model(x, y_, is_train=True)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_train')
    log_tensors = {'cost': cost}
    return net, cost, log_tensors


def build_validation(x, y_):
    net = model(x, y_, is_train=False)
    cost = tl.cost.cross_entropy(net.outputs, y_, name='cost_test')
    correct_prediction = tf.equal(tf.argmax(net.outputs, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc_test')
    return net, [cost, acc]


if __name__ == '__main__':
    # Load MNIST data
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

    # Setup the trainer
    training_dataset = make_dataset(X_train, y_train)
    # validation_dataset = make_dataset(X_val, y_val)
    trainer = tl.distributed.Trainer(
        build_training_func=build_train,
        training_dataset=training_dataset,
        batch_size=64,
        optimizer=tf.train.RMSPropOptimizer,
        optimizer_args={'learning_rate': 0.001},
        # validation_dataset=validation_dataset, build_validation_func=build_validation
    )

    # Train the network and validate every 20 mini-batches
    # 1. Easiest way to train all data: trainer.train_to_end()
    # 2. Train with validation in the middle: trainer.train_and_validate_to_end(validate_step_size=100)
    # 3. Train with full control like follows:
    while not trainer.sess.should_stop():
        trainer.train_on_batch()
        if trainer.global_step % 500 == 0 and trainer.is_master:
            tl.files.save_npz(sess=trainer.sess, name='model.npz', save_list=trainer.training_network.all_params)

    # TODO: Test the trained model
