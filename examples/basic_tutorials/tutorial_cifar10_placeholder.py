#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def model(x_crop, y_, reuse):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(name='input')(x_crop)
        net = Conv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, name='cnn1')(net)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(net)
        net = LocalResponseNormLayer(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')(net)

        net = Conv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, name='cnn2')(net)
        net = LocalResponseNormLayer(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')(net)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(net)

        net = FlattenLayer(name='flatten')(net)
        net = DenseLayer(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d1relu')(net)
        net = DenseLayer(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d2relu')(net)
        net = DenseLayer(n_units=10, act=None, W_init=W_init2, name='output')(net)
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def model_batch_norm(x_crop, y_, reuse, is_train):
    """Batch normalization should be placed before rectifier."""
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init2 = tf.constant_initializer(value=0.1)
    with tf.variable_scope("model", reuse=reuse):
        net = InputLayer(name='input')(x_crop)
        net = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='cnn1')(net)
        net = BatchNormLayer(decay=0.99, act=tf.nn.relu, name='batch1')(net, is_train=is_train)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(net)

        net = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='cnn2')(net)
        net = BatchNormLayer(decay=0.99, act=tf.nn.relu, name='batch2')(net, is_train=is_train)
        net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(net)

        net = FlattenLayer(name='flatten')
        net = DenseLayer(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d1relu')(net)
        net = DenseLayer(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d2relu')(net)
        net = DenseLayer(n_units=10, act=None, W_init=W_init2, name='output')(net)
        y = net.outputs

        ce = tl.cost.cross_entropy(y, y_, name='cost')
        # L2 for the MLP, without this, the accuracy will be reduced by 15%.
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
        cost = ce + L2

        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return net, cost, acc


def distort_fn(x, is_train=False):
    """
    The images are processed as follows:
    .. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
    .. They are approximately whitened to make the model insensitive to dynamic range.
    For training, we additionally apply a series of random distortions to
    artificially increase the data set size:
    .. Randomly flip the image from left to right.
    .. Randomly distort the image brightness.
    """
    # print('begin',x.shape, np.min(x), np.max(x))
    x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    # print('after crop',x.shape, np.min(x), np.max(x))
    if is_train:
        # x = tl.prepro.zoom(x, zoom_range=(0.9, 1.0), is_random=True)
        # print('after zoom', x.shape, np.min(x), np.max(x))
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        # print('after flip',x.shape, np.min(x), np.max(x))
        x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
        # print('after brightness',x.shape, np.min(x), np.max(x))
        # tmp = np.max(x)
        # x += np.random.uniform(-20, 20)
        # x /= tmp
    # normalize the image
    x = (x - np.mean(x)) / max(np.std(x), 1e-5)  # avoid values divided by 0
    # print('after norm', x.shape, np.min(x), np.max(x), np.mean(x))
    return x


x = tf.placeholder(dtype=tf.float32, shape=[None, 24, 24, 3], name='x')
y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='y_')

# using local response normalization
# network, cost, _ = model(x, y_, False)
# _, cost_test, acc = model(x, y_, True)
# you may want to try batch normalization
network, cost, _ = model_batch_norm(x, y_, False, is_train=True)
_, cost_test, acc = model_batch_norm(x, y_, True, is_train=False)

# train
n_epoch = 50000
learning_rate = 0.0001
print_freq = 1
batch_size = 128

train_params = network.all_params
train_op = tf.train.AdamOptimizer(
    learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False
).minimize(
    cost, var_list=train_params
)

tl.layers.initialize_global_variables(sess)

network.print_params(False)
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

for epoch in range(n_epoch):
    start_time = time.time()
    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=True)  # data augmentation for training
        sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        # train_loss, train_acc, n_batch = 0, 0, 0
        # for X_train_a, y_train_a in tl.iterate.minibatches(
        #                         X_train, y_train, batch_size, shuffle=True):
        #     X_train_a = tl.prepro.threading_data(X_train_a, fn=distort_fn, is_train=False)  # central crop
        #     err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
        #     train_loss += err; train_acc += ac; n_batch += 1
        # print("   train loss: %f" % (train_loss/ n_batch))
        # print("   train acc: %f" % (train_acc/ n_batch))
        test_loss, test_acc, n_batch = 0, 0, 0
        for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
            X_test_a = tl.prepro.threading_data(X_test_a, fn=distort_fn, is_train=False)  # central crop
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
            test_loss += err
            test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))
