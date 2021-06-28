#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import time
import numpy as np
import multiprocessing
import tensorflow as tf

from tensorlayer.layers import Module, SequentialLayer
import tensorlayer as tl
from tensorlayer.layers import (Conv2d, Dense, Flatten, MaxPool2d, BatchNorm2d, Elementwise)

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


class Block(Module):

    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.dense1 = Dense(in_channels=in_channels, n_units=256)
        self.dense2 = Dense(in_channels=256, n_units=384)
        self.dense3 = Dense(in_channels=in_channels, n_units=384)
        self.concat = Elementwise(combine_fn=tl.ops.add)

    def forward(self, inputs):
        z = self.dense1(inputs)
        z1 = self.dense2(z)

        z2 = self.dense3(inputs)
        out = self.concat([z1, z2])
        return out


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        # weights init
        W_init = tl.initializers.truncated_normal(stddev=5e-2)
        W_init2 = tl.initializers.truncated_normal(stddev=0.04)
        b_init2 = tl.initializers.constant(value=0.1)

        self.conv1 = Conv2d(64, (5, 5), (1, 1), padding='SAME', W_init=W_init, b_init=None, name='conv1', in_channels=3)
        self.bn = BatchNorm2d(num_features=64, act=tl.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool1')

        self.conv2 = Conv2d(
            64, (5, 5), (1, 1), padding='SAME', act=tl.ReLU, W_init=W_init, b_init=None, name='conv2', in_channels=64
        )
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), padding='SAME', name='pool2')

        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(384, act=tl.ReLU, W_init=W_init2, b_init=b_init2, name='dense1relu', in_channels=2304)
        self.dense_add = self.make_layer(in_channel=384)

        self.dense2 = Dense(192, act=tl.ReLU, W_init=W_init2, b_init=b_init2, name='dense2relu', in_channels=384)
        self.dense3 = Dense(10, act=None, W_init=W_init2, name='output', in_channels=192)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense_add(z)

        z = self.dense2(z)
        z = self.dense3(z)
        return z

    def make_layer(self, in_channel):
        layers = []

        _block = Block(in_channel)
        layers.append(_block)

        for _ in range(1, 3):
            range_block = Block(in_channel)
            layers.append(range_block)

        return SequentialLayer(layers)


# get the network
net = CNN()
print(net)
# training settings
batch_size = 128
n_epoch = 500
learning_rate = 0.0001
print_freq = 5
n_step_epoch = int(len(y_train) / batch_size)
n_step = n_epoch * n_step_epoch
shuffle_buffer_size = 128

train_weights = net.trainable_weights
optimizer = tl.optimizers.Adam(learning_rate)


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
        net.set_train()

        with tf.GradientTape() as tape:
            # compute outputs
            _logits = net(X_batch)
            # compute loss and update model
            _loss_ce = tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='train_loss')

        grad = tape.gradient(_loss_ce, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

        train_loss += _loss_ce
        train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
        n_iter += 1

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

    # use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

        net.set_eval()
        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in test_ds:
            _logits = net(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

# use testing data to evaluate the model
net.set_eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in test_ds:
    _logits = net(X_batch)
    test_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test loss: {}".format(test_loss / n_iter))
print("   test acc:  {}".format(test_acc / n_iter))
