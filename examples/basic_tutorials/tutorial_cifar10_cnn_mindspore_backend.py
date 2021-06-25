#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'mindspore'

import time
import numpy as np
import multiprocessing
import tensorflow as tf
from tensorlayer.layers import Module
import tensorlayer as tl
from tensorlayer.layers import (Conv2d, Dense, Flatten, MaxPool2d, BatchNorm2d)

from mindspore.nn import Momentum, WithLossCell
from mindspore import ParameterTuple
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import composite as C
import mindspore.ops.operations as P

# enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


class CNN(Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(64, (5, 5), (2, 2), b_init=None, name='conv1', in_channels=3, act=tl.ReLU)
        self.bn = BatchNorm2d(num_features=64, act=tl.ReLU)
        self.maxpool1 = MaxPool2d((3, 3), (2, 2), name='pool1')
        self.conv2 = Conv2d(128, (5, 5), (2, 2), act=tl.ReLU, b_init=None, name='conv2', in_channels=64)
        self.maxpool2 = MaxPool2d((3, 3), (2, 2), name='pool2')

        self.flatten = Flatten(name='flatten')
        self.dense1 = Dense(120, act=tl.ReLU, name='dense1relu', in_channels=512)
        self.dense2 = Dense(84, act=tl.ReLU, name='dense2relu', in_channels=120)
        self.dense3 = Dense(10, act=None, name='output', in_channels=84)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn(z)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)
        return z


# training settings
batch_size = 128
n_epoch = 500
shuffle_buffer_size = 128

# prepare cifar10 data
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
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


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def forward(self, x, label):
        return C.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)


# dataset API and augmentation
train_ds = tf.data.Dataset.from_generator(
    generator_train, output_types=(tf.float32, tf.int32)
)  # , output_shapes=((24, 24, 3), (1)))
train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
# train_ds = train_ds.repeat(n_epoch)
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.prefetch(buffer_size=4096)
train_ds = train_ds.batch(batch_size)

# get the network
net = CNN()
train_weights = net.trainable_weights
# optimizer = Adam(train_weights, learning_rate=0.01)
optimizer = Momentum(train_weights, 0.01, 0.5)
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_with_criterion = WithLossCell(net, criterion)
train_network = GradWrap(net_with_criterion)
train_network.set_train()
# print(train_weights)
for epoch in range(n_epoch):
    start_time = time.time()
    train_network.set_train()
    train_loss, train_acc, n_iter = 0, 0, 0
    for X_batch, y_batch in train_ds:
        X_batch = ms.Tensor(X_batch.numpy(), dtype=ms.float32)
        y_batch = ms.Tensor(y_batch.numpy(), dtype=ms.int32)
        output = net(X_batch)
        loss_output = criterion(output, y_batch)
        grads = train_network(X_batch, y_batch)
        success = optimizer(grads)
        loss = loss_output.asnumpy()
        train_loss += loss
        n_iter += 1
        train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
        print(" loss ", loss)
