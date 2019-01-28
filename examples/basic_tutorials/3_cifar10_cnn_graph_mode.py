#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sys
# sys.path.append("/home/haodong2/Rundi/code/tensorlayer2")
import time
import multiprocessing
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Conv2d, BatchNorm, MaxPool2d, Flatten, Dense, LocalResponseNorm
from tensorlayer.models import Model
import numpy as np

# enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# prepare cifar10 data
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

# training settings
batch_size = 128
n_epoch = 50000
# learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128 # 100
init_learning_rate = 0.001
learning_rate_decay_factor = 0.1
num_epoch_decay = 350
n_decay_steps = num_epoch_decay * n_step_epoch


# define the network
def get_model(inputs_shape):
    # self defined initialization
    W_init = tl.initializers.truncated_normal(stddev=5e-2)
    W_init2 = tl.initializers.truncated_normal(stddev=0.04)
    b_init2 = tl.initializers.constant(value=0.1)

    # build network
    ni = Input(inputs_shape)
    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv1')(ni)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)
    nn = LocalResponseNorm(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm1")(nn)

    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', act=tf.nn.relu, W_init=W_init, b_init=None, name='conv2')(nn)
    nn = LocalResponseNorm(depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name="norm2")(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu')(nn)
    nn = Dense(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense2relu')(nn)
    nn = Dense(10, act=None, W_init=W_init2, name='output')(nn)

    M = Model(inputs=ni, outputs=nn, name='cnn')

    return M


net = get_model([None, 24, 24, 3])


def get_cost_acc(model, x, y_, is_train):
    y = model(x, is_train=is_train).outputs

    ce = tl.cost.cross_entropy(y, y_, name='cost')

    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
    cost = ce + L2

    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int64), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cost, acc


x = tf.placeholder(tf.float32, shape=[None, 24, 24, 3], name='inputs')
y_ = tf.placeholder(tf.int64, shape=[None], name='targets')

cost, acc, = get_cost_acc(net, x, y_, True)
cost_test, acc_test = get_cost_acc(net, x, y_, False)

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(init_learning_rate, global_step, n_decay_steps,
                                learning_rate_decay_factor, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, global_step=global_step)


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
    x = tl.prepro.crop_multi(x, 24, 24, is_random=is_train)
    # print('after crop',x.shape, np.min(x), np.max(x))
    if is_train:
        # x = tl.prepro.zoom(x, zoom_range=(0.9, 1.0), is_random=True)
        # print('after zoom', x.shape, np.min(x), np.max(x))
        x = tl.prepro.flip_axis_multi(x, axis=1, is_random=True)
        # print('after flip',x.shape, np.min(x), np.max(x))
        x = tl.prepro.brightness_multi(x, gamma=0.1, gain=1, is_random=True)
        # print('after brightness',x.shape, np.min(x), np.max(x))
        # tmp = np.max(x)
        # x += np.random.uniform(-20, 20)
        # x /= tmp
    # normalize the image
    x = (x - np.mean(x)) / max(np.std(x), 1e-5)  # avoid values divided by 0
    # print('after norm', x.shape, np.min(x), np.max(x), np.mean(x))
    return x


## initialize all variables in the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        X_batch_a = tl.prepro.threading_data(X_batch, fn=distort_fn, thread_count=8, is_train=True)
        _loss, _acc, _ = sess.run([cost, acc, train_op], feed_dict={x: X_batch_a, y_: y_batch})
        train_loss += _loss
        train_acc += _acc
        n_iter += 1

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))

        print("   train loss: %f" % (train_loss / n_iter))
        print("   train acc: %f" % (train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
            X_batch_a = tl.prepro.threading_data(X_batch, fn=distort_fn, thread_count=8, is_train=False)
            _loss, _acc = sess.run([cost_test, acc_test], feed_dict={x: X_batch_a, y_: y_batch})
            val_loss += _loss
            val_acc += _acc
            n_iter += 1
        print("   val loss: %f" % (val_loss / n_iter))
        print("   val acc: %f" % (val_acc / n_iter))

sess.close()
