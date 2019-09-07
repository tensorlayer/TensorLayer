#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import time

import numpy as np

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model

# enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


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


def get_model_batchnorm(inputs_shape):
    # self defined initialization
    W_init = tl.initializers.truncated_normal(stddev=5e-2)
    W_init2 = tl.initializers.truncated_normal(stddev=0.04)
    b_init2 = tl.initializers.constant(value=0.1)

    # build network
    ni = Input(inputs_shape)
    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1')(ni)
    nn = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch1')(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(nn)

    nn = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv2')(nn)
    nn = BatchNorm(decay=0.99, act=tf.nn.relu, name='batch2')(nn)
    nn = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(nn)

    nn = Flatten(name='flatten')(nn)
    nn = Dense(384, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense1relu')(nn)
    nn = Dense(192, act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='dense2relu')(nn)
    nn = Dense(10, act=None, W_init=W_init2, name='output')(nn)

    M = Model(inputs=ni, outputs=nn, name='cnn')
    return M


# get the network
net = get_model([None, 24, 24, 3])

# training settings
batch_size = 128
n_epoch = 50000
learning_rate = 0.0001
print_freq = 5
# init_learning_rate = 0.1
# learning_rate_decay_factor = 0.1
# num_epoch_decay = 350

train_weights = net.trainable_weights
# learning_rate = tf.Variable(init_learning_rate)
optimizer = tf.optimizers.Adam(learning_rate)


def _fn_train(img, target):
    # 1. Randomly crop a [height, width] section of the image.
    img = tl.prepro.crop(img, 24, 24, False)
    # # 2. Randomly flip the image horizontally.
    img = tl.prepro.flip_axis(img, is_random=True)
    # # 3. Randomly change brightness.
    # # 4. Randomly change contrast.
    # img = tl.prepro.brightness(img, is_random=True)
    # # 5. Subtract off the mean and divide by the variance of the pixels.
    img = tl.prepro.samplewise_norm(img)
    # img = tl.prepro.featurewise_norm(img)
    target = np.reshape(target, ())
    return img, target


def _fn_test(img, target):
    # 1. Crop the central [height, width] of the image.
    img = tl.prepro.crop(img, 24, 24)
    # 2. Subtract off the mean and divide by the variance of the pixels.
    img = tl.prepro.samplewise_norm(img)
    img = np.reshape(img, (24, 24, 3))
    target = np.reshape(target, ())
    return img, target


# dataset API and augmentation
train_ds = tl.data.CIFAR10(train_or_test='train', shape=(-1, 32, 32, 3))
train_dl = tl.data.Dataloader(train_ds, transforms=[_fn_train], shuffle=True,
                              batch_size=batch_size, output_types=(np.float32, np.int32))
test_ds = tl.data.CIFAR10(train_or_test='test', shape=(-1, 32, 32, 3))
test_dl = tl.data.Dataloader(test_ds, transforms=[_fn_test], batch_size=batch_size)

for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in train_dl:
        net.train()

        with tf.GradientTape() as tape:
            # compute outputs
            _logits = net(X_batch)
            # compute loss and update model
            _loss_ce = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
            _loss_L2 = 0
            # for p in tl.layers.get_variables_with_name('relu/W', True, True):
            #      _loss_L2 += tl.cost.lo_regularizer(1.0)(p)
            _loss = _loss_ce + _loss_L2
            # print(_loss)

        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

        train_loss += _loss
        train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
        n_iter += 1

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))

        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        net.eval()

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_dl:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

    # FIXME : how to apply lr decay in eager mode?
    # learning_rate.assign(tf.train.exponential_decay(init_learning_rate, epoch, num_epoch_decay,
    #                                                 learning_rate_decay_factor))

# use testing data to evaluate the model
net.eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in test_dl:
    _logits = net(X_batch)
    test_loss += tl.cost.cross_entropy(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test loss: {}".format(test_loss / n_iter))
print("   test acc:  {}".format(test_acc / n_iter))
