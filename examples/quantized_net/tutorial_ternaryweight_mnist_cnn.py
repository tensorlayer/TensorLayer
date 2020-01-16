#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import (BatchNorm, Dense, Flatten, Input, MaxPool2d, TernaryConv2d, TernaryDense)
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

batch_size = 128


def model(inputs_shape, n_class=10):
    in_net = Input(inputs_shape, name='input')
    net = TernaryConv2d(32, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn1')(in_net)
    net = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')(net)
    net = BatchNorm(act=tl.act.htanh, name='bn1')(net)

    net = TernaryConv2d(64, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn2')(net)
    net = MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')(net)
    net = BatchNorm(act=tl.act.htanh, name='bn2')(net)

    net = Flatten('flatten')(net)
    net = Dense(256, b_init=None, name='dense')(net)
    net = BatchNorm(act=tl.act.htanh, name='bn3')(net)

    net = TernaryDense(n_class, b_init=None, name='bout')(net)
    net = BatchNorm(name='bno')(net)

    net = Model(inputs=in_net, outputs=net, name='dorefanet')
    return net


def _train_step(network, X_batch, y_batch, cost, train_op=tf.optimizers.Adam(learning_rate=0.0001), acc=None):
    with tf.GradientTape() as tape:
        y_pred = network(X_batch)
        _loss = cost(y_pred, y_batch)
    grad = tape.gradient(_loss, network.trainable_weights)
    train_op.apply_gradients(zip(grad, network.trainable_weights))
    if acc is not None:
        _acc = acc(y_pred, y_batch)
        return _loss, _acc
    else:
        return _loss, None


def accuracy(_logits, y_batch):
    return np.mean(np.equal(np.argmax(_logits, 1), y_batch))


n_epoch = 200
print_freq = 5

net = model([None, 28, 28, 1])
train_op = tf.optimizers.Adam(learning_rate=0.0001)
cost = tl.cost.cross_entropy

for epoch in range(n_epoch):
    start_time = time.time()
    train_loss, train_acc, n_batch = 0, 0, 0
    net.train()

    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        _loss, acc = _train_step(net, X_train_a, y_train_a, cost=cost, train_op=train_op, acc=accuracy)
        train_loss += _loss
        train_acc += acc
        n_batch += 1

        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
        print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        val_loss, val_acc, val_batch = 0, 0, 0
        net.eval()
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
            _logits = net(X_val_a)
            val_loss += tl.cost.cross_entropy(_logits, y_val_a, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_val_a))
            val_batch += 1
        print("   val loss: {}".format(val_loss / val_batch))
        print("   val acc:  {}".format(val_acc / val_batch))

net.test()
test_loss, test_acc, n_test_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    _logits = net(X_test_a)
    test_loss += tl.cost.cross_entropy(_logits, y_test_a, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_test_a))
    n_test_batch += 1
print("   test loss: %f" % (test_loss / n_test_batch))
print("   test acc: %f" % (test_acc / n_test_batch))
