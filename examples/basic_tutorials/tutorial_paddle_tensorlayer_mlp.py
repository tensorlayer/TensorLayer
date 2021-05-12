#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'paddle'

from paddle.vision.transforms import Compose, Normalize
import paddle

import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Flatten

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

class MLP(Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = Dense(n_units=120, in_channels=784, act=tl.ReLU)
        self.linear2 = Dense(n_units=84, in_channels=120, act=tl.ReLU)
        self.linear3 = Dense(n_units=10, in_channels=84)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

net = MLP()
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=net.trainable_weights)
model = tl.models.Model(network=net, loss_fn=tl.cost.cross_entropy, optimizer=optimizer)
model.train(n_epoch=20, train_dataset=train_loader, print_freq=5, print_train_batch=True)
