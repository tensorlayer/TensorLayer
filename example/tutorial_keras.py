#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time
from keras import backend as K
from keras.layers import *
from tensorlayer.layers import *

X_train, y_train, X_val, y_val, X_test, y_test = \
                tl.files.load_mnist_dataset(shape=(-1, 784))

sess = tf.InteractiveSession()

batch_size = 128
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int64, shape=[None,])

def keras_block(x):
    x = Dropout(0.8)(x)
    x = Dense(800, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(800, activation='relu')(x)
    x = Dropout(0.5)(x)
    logits = Dense(10, activation='linear')(x)
    return logits

network = InputLayer(x, name='input')
network = KerasLayer(network, keras_layer=keras_block, name='keras')

y = network.outputs
network.print_params(False)
network.print_layers()

cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

n_epoch = 200
learning_rate = 0.0001

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

for epoch in range(n_epoch):
    start_time = time.time()
    ## Training
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        _, _ = sess.run([cost, train_op], feed_dict={x: X_train_a, y_: y_train_a,
                                K.learning_phase(): 1})

    print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
    ## Evaluation
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_a, y_train_a in tl.iterate.minibatches(
                            X_train, y_train, batch_size, shuffle=False):
        err, ac = sess.run([cost, acc], feed_dict={x: X_train_a, y_: y_train_a,
                                K.learning_phase(): 0})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (train_loss/ n_batch))
    print("   train acc: %f" % (train_acc/ n_batch))
    val_loss, val_acc, n_batch = 0, 0, 0
    for X_val_a, y_val_a in tl.iterate.minibatches(
                                X_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([cost, acc], feed_dict={x: X_val_a, y_: y_val_a,
                                K.learning_phase(): 0})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   val loss: %f" % (val_loss/ n_batch))
    print("   val acc: %f" % (val_acc/ n_batch))
