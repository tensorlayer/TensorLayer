#! /usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TL_BACKEND'] = 'tensorflow'

from tensorlayer.layers import SequentialLayer
from tensorlayer.layers import Dense
import tensorlayer as tl
import numpy as np

layer_list = []
layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=784, name='Dense1'))
layer_list.append(Dense(n_units=800, act=tl.ReLU, in_channels=800, name='Dense2'))
layer_list.append(Dense(n_units=10, act=tl.ReLU, in_channels=800, name='Dense3'))
MLP = SequentialLayer(layer_list)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


def generator_train():
    inputs = X_train
    targets = y_train
    if len(inputs) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")
    for _input, _target in zip(inputs, targets):
        yield (_input, np.array(_target))


n_epoch = 50
batch_size = 128
print_freq = 2
shuffle_buffer_size = 128

# train_weights = MLP.trainable_weights
# print(train_weights)
optimizer = tl.optimizers.Momentum(0.05, 0.9)
train_ds = tl.dataflow.FromGenerator(
    generator_train, output_types=(tl.float32, tl.int32), column_names=['data', 'label']
)
train_ds = tl.dataflow.Shuffle(train_ds, shuffle_buffer_size)
train_ds = tl.dataflow.Batch(train_ds, batch_size)

model = tl.models.Model(network=MLP, loss_fn=tl.cost.softmax_cross_entropy_with_logits, optimizer=optimizer)
model.train(n_epoch=n_epoch, train_dataset=train_ds, print_freq=print_freq, print_train_batch=False)
model.save_weights('./model.npz', format='npz_dict')
model.load_weights('./model.npz', format='npz_dict')
