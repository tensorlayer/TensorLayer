#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'tensorflow'

import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Flatten

print('download training data and load training data')

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))

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


traindataset = tl.dataflow.FromSlices((X_train, y_train))
train_loader = tl.dataflow.Dataloader(traindataset, batch_size=64, shuffle=True)
net = MLP()

optimizer = tl.optimizers.Adam(learning_rate=0.001)
metric = tl.metric.Accuracy()
model = tl.models.Model(network=net, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer, metrics=metric)
model.train(n_epoch=2, train_dataset=train_loader, print_freq=5, print_train_batch=True)
model.save_weights('./model_mlp.npz', format='npz_dict')
model.load_weights('./model_mlp.npz', format='npz_dict')
# model.eval(train_loader)