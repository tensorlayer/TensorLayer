#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

- 1. This model has 1,068,298 paramters and quantization compression strategy(weight:8 bits, active: 8 bits here, you can change the setting),
after 705 epoches' training with GPU, test accurcy of 84.0% was found.

- 2. For simplified CNN layers see "Convolutional layer (Simplified)"
in read the docs website.

- 3. Data augmentation without TFRecord see `tutorial_image_preprocess.py` !!

Links
-------
.. paper:https://arxiv.org/abs/1712.05877


Note
------
The optimizers between official code and this code are different.

Description
-----------
The images are processed as follows:
.. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
.. They are approximately whitened to make the model insensitive to dynamic range.

For training, we additionally apply a series of random distortions to
artificially increase the data set size:
.. Randomly flip the image from left to right.
.. Randomly distort the image brightness.
.. Randomly distort the image contrast.

Speed Up
--------
Reading images from disk and distorting them can use a non-trivial amount
of processing time. To prevent these operations from slowing down training,
we run them inside 16 separate threads which continuously fill a TensorFlow queue.

"""
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl

bitW = 8
bitA = 8

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def model(x, y_, reuse, is_train, bitW, bitA):
    with tf.variable_scope("model", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.QuanConv2dWithBN(
            net, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', is_train=is_train, bitW=bitW, bitA=bitA,
            name='qcnnbn1'
        )
        net = tl.layers.MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool1')
        # net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn1')
        net = tl.layers.QuanConv2dWithBN(
            net, 64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, is_train=is_train, bitW=bitW, bitA=bitA,
            name='qcnnbn2'
        )
        # net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn2')
        net = tl.layers.MaxPool2d(net, (3, 3), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.FlattenLayer(net, name='flatten')
        net = tl.layers.QuanDenseLayer(net, 384, act=tf.nn.relu, bitW=bitW, bitA=bitA, name='qd1relu')
        net = tl.layers.QuanDenseLayer(net, 192, act=tf.nn.relu, bitW=bitW, bitA=bitA, name='qd2relu')
        net = tl.layers.DenseLayer(net, 10, act=None, name='output')
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int64), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def distort_fn(x, is_train=False):
    x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    if is_train:
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
    x = (x - np.mean(x)) / max(np.std(x), 1e-5)  # avoid values divided by 0
    return x


x = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 3], name='x')
y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='y_')

network, cost, _ = model(x, y_, False, True, bitW=bitW, bitA=bitA)
_, cost_test, acc = model(x, y_, True, False, bitW=bitW, bitA=bitA)

# train
n_epoch = 50000
learning_rate = 0.0001
print_freq = 1
batch_size = 128

train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                  use_locking=False).minimize(cost)

sess.run(tf.global_variables_initializer())

network.print_params(False)
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)
print('   bitW: %d,   bitA: %d' % (bitW, bitA))

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=True)  # data augmentation for training
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
            X_test_a = tl.prepro.threading_data(X_test_a, fn=distort_fn, is_train=False)  # central crop
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
            test_loss += err
            test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))
