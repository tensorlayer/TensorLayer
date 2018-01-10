#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

D_TYPE = tf.float16         # tf.float32  tf.float16
tl.layers.D_TYPE = D_TYPE   # define global dtype in tl.layers

X_train, y_train, X_val, y_val, X_test, y_test = \
                tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(D_TYPE, shape=[batch_size, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[batch_size,])

def model(x, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(x, name='input')
        # cnn
        n = Conv2d(n, 32, (5, 5), (1, 1), padding='SAME', name='cnn1')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool1')
        n = Conv2d(n, 64, (5, 5), (1, 1), padding='SAME', name='cnn2')
        n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, name='bn2')
        n = MaxPool2d(n, (2, 2), (2, 2), padding='SAME', name='pool2')
        # mlp
        n = FlattenLayer(n, name='flatten')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop1')
        n = DenseLayer(n, 256, act=tf.nn.relu, name='relu1')
        n = DropoutLayer(n, 0.5, True, is_train, name='drop2')
        n = DenseLayer(n, 10, act=tf.identity, name='output')
    return n

# define inferences
net_train = model(x, is_train=True, reuse=False)
net_test = model(x, is_train=False, reuse=True)

net_train.print_params(False)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, D_TYPE))

# define the optimizer
train_params = tl.layers.get_variables_with_name('model', train_only=True, printable=False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                # epsilon=1e-08,    # for float32 as default
                epsilon=1e-4,       # for float16, see https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
                use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# train the network
n_epoch = 500
print_freq = 1

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                X_train, y_train, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (train_loss/ n_batch))
        print("   train acc: %f" % (train_acc/ n_batch))
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(
                                    X_val, y_val, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
            val_loss += err; val_acc += ac; n_batch += 1
        print("   val loss: %f" % (val_loss/ n_batch))
        print("   val acc: %f" % (val_acc/ n_batch))

print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
                            X_test, y_test, batch_size, shuffle=True):
    err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err; test_acc += ac; n_batch += 1
print("   test loss: %f" % (test_loss/n_batch))
print("   test acc: %f" % (test_acc/n_batch))
