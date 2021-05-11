#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'paddle'

import paddle.nn.functional as F
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

def train(model):
    model.train()
    epochs = 2
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.trainable_weights)
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 50 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()
model = MLP()
train(model)

