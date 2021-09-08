#! /usr/bin/python
# -*- coding: utf-8 -*-
"""

- 1. This model has 1,068,298 paramters and TWN compression strategy(weight:1,0,-1, output: float32),
after 500 epoches' training with GPU,accurcy of 80.6% was found.

- 2. For simplified CNN layers see "Convolutional layer (Simplified)"
in read the docs website.

- 3. Data augmentation without TFRecord see `tutorial_image_preprocess.py` !!

Links
-------
.. https://arxiv.org/abs/1605.04711
.. https://github.com/XJTUWYD/TWN

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
import multiprocessing
import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import (
    Conv2d, Dense, Flatten, Input, LocalResponseNorm, MaxPool2d, TernaryConv2d, TernaryDense
)
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)

# Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
# prepare cifar10 data
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def model(input_shape, n_classes):
    in_net = Input(shape=input_shape, name='input')

    net = Conv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')(in_net)
    net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')(net)
    net = LocalResponseNorm(4, 1.0, 0.001 / 9.0, 0.75, name='norm1')(net)

    net = TernaryConv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')(net)
    net = LocalResponseNorm(4, 1.0, 0.001 / 9.0, 0.75, name='norm2')(net)
    net = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')(net)

    net = Flatten(name='flatten')(net)

    net = TernaryDense(384, act=tf.nn.relu, name='d1relu')(net)
    net = TernaryDense(192, act=tf.nn.relu, name='d2relu')(net)
    net = Dense(n_classes, act=None, name='output')(net)

    net = Model(inputs=in_net, outputs=net, name='dorefanet')
    return net


# training settings
bitW = 8
bitA = 8
net = model([None, 24, 24, 3], n_classes=10)
batch_size = 128
n_epoch = 50000
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

optimizer = tf.optimizers.Adam(learning_rate)
cost = tl.cost.cross_entropy


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


def _train_step(network, X_batch, y_batch, cost, train_op=tf.optimizers.Adam(learning_rate=0.0001), acc=None):
    with tf.GradientTape() as tape:
        y_pred = network(X_batch)
        _loss = cost(y_pred, y_batch)
    grad = tape.gradient(_loss, network.trainable_weights)
    train_op.apply_gradients(zip(grad, network.trainable_weights))
    if acc is not None:
        _acc = acc(y_pred, y_batch)
        return _loss, _acc
    else:
        return _loss, None


def accuracy(_logits, y_batch):
    return np.mean(np.equal(np.argmax(_logits, 1), y_batch))


# dataset API and augmentation
train_ds = tf.data.Dataset.from_generator(
    generator_train, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
# train_ds = train_ds.repeat(n_epoch)
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.prefetch(buffer_size=4096)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
# value = train_ds.make_one_shot_iterator().get_next()

test_ds = tf.data.Dataset.from_generator(
    generator_test, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
# test_ds = test_ds.shuffle(shuffle_buffer_size)
# test_ds = test_ds.repeat(n_epoch)
test_ds = test_ds.prefetch(buffer_size=4096)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.map(_map_fn_test, num_parallel_calls=multiprocessing.cpu_count())
# value_test = test_ds.make_one_shot_iterator().get_next()

for epoch in range(n_epoch):
    start_time = time.time()

    train_loss, train_acc, n_iter = 0, 0, 0
    net.train()
    for X_batch, y_batch in train_ds:
        _loss, acc = _train_step(net, X_batch, y_batch, cost=cost, train_op=optimizer, acc=accuracy)

        train_loss += _loss
        train_acc += acc
        n_iter += 1

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        net.eval()
        val_loss, val_acc, n_val_iter = 0, 0, 0
        for X_batch, y_batch in test_ds:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_val_iter += 1
        print("   val loss: {}".format(val_loss / n_val_iter))
        print("   val acc:  {}".format(val_acc / n_val_iter))

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
