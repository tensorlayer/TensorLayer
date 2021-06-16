#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The same set of code can switch the backend with one line
import os
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TL_BACKEND'] = 'mindspore'

import numpy as np
import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Dropout

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(keep=0.8)
        self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
        self.dropout2 = Dropout(keep=0.8)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dropout3 = Dropout(keep=0.8)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.dense1(z)
        # z = self.bn(z)
        z = self.dropout2(z)
        z = self.dense2(z)
        z = self.dropout3(z)
        out = self.dense3(z)
        if foo is not None:
            out = tl.ops.relu(out)
        return out


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield (_input, np.array(_target))


MLP = CustomModel()

n_epoch = 50
batch_size = 128
print_freq = 2
shuffle_buffer_size = 128

train_weights = MLP.trainable_weights
optimizer = tl.optimizers.Momentum(0.05, 0.9)
train_ds = tl.dataflow.FromGenerator(
    generator_train, output_types=(tl.float32, tl.int32) , column_names=['data', 'label']
)
train_ds = tl.dataflow.Shuffle(train_ds,shuffle_buffer_size)
train_ds = tl.dataflow.Batch(train_ds,batch_size)


model = tl.models.Model(network=MLP, loss_fn=tl.cost.cross_entropy, optimizer=optimizer)
model.train(n_epoch=n_epoch, train_dataset=train_ds, print_freq=print_freq, print_train_batch=False)
model.save_weights('./model.npz', format='npz_dict')
model.load_weights('./model.npz', format='npz_dict')
