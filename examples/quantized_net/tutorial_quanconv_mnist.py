#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

import tensorlayer as tl
from tensorlayer.layers import (
    Dense, Dropout, Flatten, Input, MaxPool2d, QuanConv2d, QuanConv2dWithBN, QuanDense, QuanDenseLayerWithBN
)
from tensorlayer.models import Model

tl.logging.set_verbosity(tl.logging.DEBUG)

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

batch_size = 128


def model(inputs_shape, n_class=10):
    net_in = Input(inputs_shape, name="input")

    net = QuanConv2dWithBN(
        n_filter=32, filter_size=(5, 5), strides=(1, 1), padding='SAME', act=tl.nn.relu, name='qconvbn1'
    )(net_in)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')(net)

    net = QuanConv2dWithBN(
        n_filter=64, filter_size=(5, 5), strides=(1, 1), padding='SAME', act=tl.nn.relu, name='qconvbn2'
    )(net)
    net = MaxPool2d(filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')(net)

    net = Flatten(name='ft')(net)

    # net = QuanDense(256, act="relu", name='qdbn')(net)
    # net = QuanDense(n_class, name='qdbn_out')(net)

    net = QuanDenseLayerWithBN(256, act="relu", name='qdbn')(net)
    net = QuanDenseLayerWithBN(n_class, name='qdbn_out')(net)

    # net = Dense(256, act='relu', name='Dense1')(net)
    # net = Dense(n_class, name='Dense2')(net)

    net = Model(inputs=net_in, outputs=net, name='quan')
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
print_freq = 1

# print(sess.run(net_test.all_params)) # print real values of parameters
net = model([None, 28, 28, 1])
train_op = tf.optimizers.Adam(learning_rate=0.0001)
cost = tl.cost.cross_entropy

for epoch in range(n_epoch):
    start_time = time.time()
    train_loss, train_acc, n_iter = 0, 0, 0

    for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
        net.train()
        _loss, acc = _train_step(net, X_train_a, y_train_a, cost=cost, train_op=train_op, acc=accuracy)

        train_loss += _loss
        train_acc += acc
        n_iter += 1

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

    if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:

        print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
        print("   train loss: {}".format(train_loss / n_iter))
        print("   train acc:  {}".format(train_acc / n_iter))

        # net.eval()
        val_loss, val_acc, n_eval = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
            _logits = net(X_val_a)  # is_train=False, disable dropout
            val_loss += tl.cost.cross_entropy(_logits, y_val_a, name='eval_loss')
            val_acc += np.mean(np.equal(np.argmax(_logits, 1), y_val_a))
            n_eval += 1
        print("   val loss: {}".format(val_loss / n_eval))
        print("   val acc:  {}".format(val_acc / n_eval))

# net.eval()
test_loss, test_acc, n_test_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    _logits = net(X_test_a)
    test_loss += tl.cost.cross_entropy(_logits, y_test_a, name='test_loss')
    test_acc += np.mean(np.equal(np.argmax(_logits, 1), y_test_a))
    n_test_batch += 1
print("   test loss: %f" % (test_loss / n_test_batch))
print("   test acc: %f" % (test_acc / n_test_batch))
