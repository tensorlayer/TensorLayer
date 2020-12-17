#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import Dense, WithLossCell, SoftmaxCrossEntropyWithLogits, Momentum
import tensorlayer as tl
import mindspore as ms
import tensorflow as tf
import time

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """

    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.fc1 = fc_with_initialize(784, 800)
        self.fc2 = fc_with_initialize(800, 800)
        self.fc3 = fc_with_initialize(800, self.num_class)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GradWrap(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return C.GradOperation('get_by_list', get_by_list=True)(self.network, weights)(x, label)


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield _input, _target


net = LeNet5()
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
net_with_criterion = WithLossCell(net, criterion)
train_network = GradWrap(net_with_criterion)
train_network.set_train()

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.int32))
shuffle_buffer_size = 128
batch_size = 128
train_ds = train_ds.shuffle(shuffle_buffer_size)
train_ds = train_ds.batch(batch_size)
n_epoch = 50

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
        # train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        # print("   train acc:  {}".format(train_acc / n_iter))
        print(" triain weights ", train_network.trainable_params()[0].data)
