#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'mindspore'

import mindspore.ops.operations as P
from mindspore.ops import composite as C
from mindspore import ParameterTuple
from mindspore.nn import Momentum, WithLossCell

import numpy as np
import tensorlayer as tl
import mindspore as ms
import tensorflow as tf
import time
from tensorlayer.layers import Module
from tensorlayer.layers import Dense
import mindspore.nn as nn


class MLP(Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x):
        z = self.dense1(x)
        z = self.dense2(z)
        out = self.dense3(z)
        return out


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def forward(self, x, label):
        return C.GradOperation(get_by_list=True)(self.network, self.weights)(x, label)


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield _input, _target


net = MLP()
train_weights = list(filter(lambda x: x.requires_grad, net.get_parameters()))
optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.15, 0.8)

criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
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
        train_acc += np.mean((P.Equal()(P.Argmax(axis=1)(output), y_batch).asnumpy()))
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))
        print(" loss ", loss)
