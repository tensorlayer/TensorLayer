#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

sess = tf.InteractiveSession()

batch_size = 128
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int64, shape=[None])
is_training = tf.placeholder(tf.bool)


def slim_block(x):
    with tf.variable_scope('tf_slim'):
        x = slim.dropout(x, 0.8, is_training=is_training)
        x = slim.fully_connected(x, 800, activation_fn=tf.nn.relu)
        x = slim.dropout(x, 0.5, is_training=is_training)
        x = slim.fully_connected(x, 800, activation_fn=tf.nn.relu)
        x = slim.dropout(x, 0.5, is_training=is_training)
        logits = slim.fully_connected(x, 10, activation_fn=None)
    return logits, {}


network = Input(name='input')(x)
network = SlimNets(slim_layer=slim_block, name='tf_slim')(network)

y = network.outputs
network.print_weights(False)
network.print_layers()

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

n_epoch = 200
learning_rate = 0.0001

train_weights = network.all_weights
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_weights)

tl.layers.initialize_global_variables(sess)

for epoch in range(n_epoch):
    start_time = time.time()
    ## Training
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        _, _ = sess.run([cost, train_op], feed_dict={x: X_train_a, y_: y_train_a, is_training: True})

    print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
    ## Evaluation
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
        err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a, is_training: False})
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))
    val_loss, val_acc, n_batch = 0, 0, 0
    for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([cost, acc], feed_dict={x: X_val_a, y_: y_val_a, is_training: False})
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   val loss: %f" % (val_loss / n_batch))
    print("   val acc: %f" % (val_acc / n_batch))
