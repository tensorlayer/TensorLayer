#! /usr/bin/python
# -*- coding: utf-8 -*-
# The tensorlayer and tensorflow operators can be mixed
import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import time

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Module
from tensorlayer.layers import Dense, Dropout, BatchNorm1d

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dropout1 = Dropout(keep=0.8)
        self.dense1 = Dense(n_units=800, in_channels=784)
        self.batchnorm = BatchNorm1d(act=tl.ReLU, num_features=800)
        self.dropout2 = Dropout(keep=0.8)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dropout3 = Dropout(keep=0.8)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dropout1(x)
        z = self.dense1(z)
        z = self.batchnorm(z)
        z = self.dropout2(z)
        z = self.dense2(z)
        z = self.dropout3(z)
        out = self.dense3(z)
        if foo is not None:
            out = tl.ops.relu(out)
        return out


MLP = CustomModel()
n_epoch = 50
batch_size = 500
print_freq = 5
train_weights = MLP.trainable_weights
optimizer = tl.optimizers.Adam(lr=0.0001)

for epoch in range(n_epoch):  ## iterate the dataset n_epoch times
    start_time = time.time()
    ## iterate over the entire training set once (shuffle the data via training)
    for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        MLP.set_train()  # enable dropout
        with tf.GradientTape() as tape:
            ## compute outputs
            _logits = MLP(X_batch)
            ## compute loss and update model
            _loss = tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='train_loss')
        grad = tape.gradient(_loss, train_weights)
        optimizer.apply_gradients(zip(grad, train_weights))

    ## use training and evaluation sets to evaluate the model every print_freq epoch
    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=False):
            _logits = MLP(X_batch)
            train_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            train_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        val_loss, val_acc, n_iter = 0, 0, 0
        for X_batch, y_batch in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=False):
            _logits = MLP(X_batch)  # is_train=False, disable dropout
            val_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            n_iter += 1
        print("   val loss: {}".format(val_loss / n_iter))
        print("   val acc:  {}".format(val_acc / n_iter))

## use testing data to evaluate the model
MLP.set_eval()
test_loss, test_acc, n_iter = 0, 0, 0
for X_batch, y_batch in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=False):
    _logits = MLP(X_batch, foo=1)
    test_loss += tl.cost.softmax_cross_entropy_with_logits(_logits, y_batch, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_batch))
    n_iter += 1
print("   test foo=1 loss: {}".format(test_loss / n_iter))
print("   test foo=1 acc:  {}".format(test_acc / n_iter))
