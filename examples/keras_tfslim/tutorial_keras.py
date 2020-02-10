#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import Input, Lambda

tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

batch_size = 128

# keras layers
layers = [
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(800, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='linear')
]
keras_block = tf.keras.Sequential(layers)
# in order to compile keras model and get trainable_variables of the keras model
_ = keras_block(np.random.random([batch_size, 784]).astype(np.float32))

# build tl model using keras layers
ni = Input([None, 784], dtype=tf.float32)
nn = Lambda(fn=keras_block, fn_weights=keras_block.trainable_variables)(ni)
network = tl.models.Model(inputs=ni, outputs=nn)
print(network)

n_epoch = 200
learning_rate = 0.0001

train_params = network.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate)

for epoch in range(n_epoch):
    start_time = time.time()
    ## Training
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        with tf.GradientTape() as tape:
            _logits = network(X_train_a, is_train=True)
            err = tl.cost.cross_entropy(_logits, y_train_a, name='train_loss')

        grad = tape.gradient(err, train_params)
        optimizer.apply_gradients(zip(grad, train_params))
        # _, _ = sess.run([cost, train_op], feed_dict={x: X_train_a, y_: y_train_a, K.learning_phase(): 1})

    print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

    ## Evaluation
    train_loss, train_acc, n_batch = 0, 0, 0
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
        _logits = network(X_train_a, is_train=False)
        err = tl.cost.cross_entropy(_logits, y_train_a, name='train_loss')
        ac = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(_logits, 1), y_train_a), tf.float32))
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))
    val_loss, val_acc, n_batch = 0, 0, 0
    for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
        _logits = network(X_val_a, is_train=False)
        err = tl.cost.cross_entropy(_logits, y_val_a, name='train_loss')
        ac = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(_logits, 1), y_val_a), tf.float32))
        val_loss += err
        val_acc += ac
        n_batch += 1
    print("   val loss: %f" % (val_loss / n_batch))
    print("   val acc: %f" % (val_acc / n_batch))
