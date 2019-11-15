#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import multiprocessing
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d)
from tensorlayer.models import Model

# enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# prepare cifar10 data
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


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
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

train_weights = net.trainable_weights
optimizer = tf.optimizers.Adam(learning_rate)
# looking for decay learning rate? see https://github.com/tensorlayer/srgan/blob/master/train.py


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        # yield _input.encode('utf-8'), _target.encode('utf-8')
        yield _input, _target


def generator_test():
    inputs = X_test
    targets = y_test
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        # yield _input.encode('utf-8'), _target.encode('utf-8')
        yield _input, _target


def _map_fn_train(img, target):
    # 1. Randomly crop a [height, width] section of the image.
    img = tf.image.random_crop(img, [24, 24, 3])
    # 2. Randomly flip the image horizontally.
    img = tf.image.random_flip_left_right(img)
    # 3. Randomly change brightness.
    img = tf.image.random_brightness(img, max_delta=63)
    # 4. Randomly change contrast.
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    # 5. Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    target = tf.reshape(target, ())
    return img, target


def _map_fn_test(img, target):
    # 1. Crop the central [height, width] of the image.
    img = tf.image.resize_with_pad(img, 24, 24)  
    # 2. Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    img = tf.reshape(img, (24, 24, 3))
    target = tf.reshape(target, ())
    return img, target


# dataset API and augmentation
train_ds = tf.data.Dataset.from_generator(
    generator_train, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
# train_ds = train_ds.repeat(n_epoch)
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.prefetch(buffer_size=4096)
train_ds = train_ds.batch(batch_size)
# value = train_ds.make_one_shot_iterator().get_next()

test_ds = tf.data.Dataset.from_generator(
    generator_test, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
# test_ds = test_ds.shuffle(shuffle_buffer_size)
test_ds = test_ds.map(_map_fn_test, num_parallel_calls=multiprocessing.cpu_count())
# test_ds = test_ds.repeat(n_epoch)
test_ds = test_ds.prefetch(buffer_size=4096)
test_ds = test_ds.batch(batch_size)
# value_test = test_ds.make_one_shot_iterator().get_next()

for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in train_ds:
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
        for X_batch, y_batch in test_ds:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

# use testing data to evaluate the model
net.eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in test_ds:
    _logits = net(X_batch)
    test_loss += tl.cost.cross_entropy(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test loss: {}".format(test_loss / n_iter))
print("   test acc:  {}".format(test_acc / n_iter))
