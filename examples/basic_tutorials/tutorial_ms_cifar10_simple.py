#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import os
# os.environ['TL_BACKEND'] = 'tensorflow'
os.environ['TL_BACKEND'] = 'mindspore'

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Dropout

from mindspore.common import ParameterTuple
import mindspore as ms
import mindspore.dataset as ds
from mindspore.ops import composite
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Momentum, TrainOneStepCell, WithLossCell
from mindspore.parallel._utils import (_get_device_num, _get_mirror_mean, _get_parallel_mode)
from mindspore.train.parallel_utils import ParallelMode
from mindspore.nn.wrap import DistributedGradReducer
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(keep=0.8)
        self.dense1 = Dense(n_units=800, in_channels=784, act=tl.ReLU)
        self.dropout2 = Dropout(keep=0.8)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dropout3 = Dropout(keep=0.8)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.dense1(z)
        z = self.dropout2(z)
        z = self.dense2(z)
        z = self.dropout3(z)
        out = self.dense3(z)
        if foo is not None:
            out = tl.ops.relu(out)
        return out


class WithLoss(Module):

    def __init__(self, backbone, loss_fn):
        super(WithLoss, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)

    @property
    def backbone_network(self):
        return self._backbone


class GradWrap(Module):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.trainable = True
        self.weights = ParameterTuple(network.trainable_weights)

    def construct(self, x, label):
        weights = self.weights
        return composite.GradOperation('get_by_list', get_by_list=True)(self.network, weights)(x, label)


class TrainOneStep(Module):

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStep, self).__init__(auto_prefix=False)
        self._built = True
        self.trainable = True
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = composite.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_mirror_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield _input, _target


MLP = CustomModel()
train_weights = MLP.trainable_weights

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32, tf.int32))

opt = Momentum(train_weights, 0.01, 0.9)
n_epoch = 50
batch_size = 128
print_freq = 2
model = tl.models.Model(network=MLP, loss_fn=SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True), optimizer=opt)
model.train(n_epoch=n_epoch, train_dataset=train_ds, print_freq=print_freq, print_train_batch=False)

# batch_size = 128
# epoch = 50
#
# # loss function definition
# ls = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
# # optimization definition
# opt = Momentum(train_weights, 0.01, 0.9)
# net_with_criterion = WithLoss(MLP, ls)
# # train_network = TrainOneStep(net_with_criterion, opt)  # optimizer
# train_network = GradWrap(net_with_criterion)
# acc = ms.nn.Accuracy()
#
# for epoch in range(epoch):
#     MLP.set_train()
#     for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
#         X_batch = ms.Tensor(X_batch, dtype=ms.float32)
#         y_batch = ms.Tensor(y_batch, dtype=ms.int32)
#         output = MLP(X_batch)
#         loss_output = ls(output, y_batch)
#         grads = train_network(X_batch, y_batch)
#         success = opt(grads)
#         loss = loss_output.asnumpy()
#         accutacy = acc()
#         print(loss)
