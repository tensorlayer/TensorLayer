#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TL_BACKEND'] = 'dragon'

from tensorlayer.layers import Module
from tensorlayer.layers import Dense
import tensorlayer as tl
import dragon as dg
import time
import argparse
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))


class CustomModel(Module):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense1 = Dense(n_units=800, act=tl.ReLU, in_channels=784)
        self.dense2 = Dense(n_units=800, act=tl.ReLU, in_channels=800)
        self.dense3 = Dense(n_units=10, act=tl.ReLU, in_channels=800)

    def forward(self, x, foo=None):
        z = self.dense1(x)
        z = self.dense2(z)
        out = self.dense3(z)
        return out


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description='Train a cifar10 resnet')
    parser.add_argument('--execution', default='EAGER_MODE', type=str, help='The execution mode')
    parser.add_argument('--seed', default=1337, type=int, help='The random seed')
    parser.add_argument('--cuda', default=-1, type=int, help='The cuda device to use')
    return parser.parse_args()


class Classifier(object):
    """The base classifier class."""

    # TensorSpec for graph execution
    image_spec = dg.Tensor([None, 3, 32, 32], 'float32')
    label_spec = dg.Tensor([None], 'int64')

    def __init__(self, optimizer):
        super(Classifier, self).__init__()
        self.net = CustomModel()
        self.optimizer = optimizer
        self.params = self.net.trainable_weights

    def step(self, image, label):
        with dg.GradientTape() as tape:
            logit = self.net(image)
            # logit = dg.cast(logit, 'float64')
            logit = dg.cast(dg.math.argmax(logit, -1), 'int64')
            label = dg.cast(label, 'int64')
            # print("logit :\n", logit, label)
            # loss = dg.losses.smooth_l1_loss([logit, label])
            loss = dg.math.sum(logit - label)  # dg.losses.sparse_softmax_cross_entropy([logit, label])
        accuracy = dg.math.mean(dg.math.equal([logit, label]).astype('float32'))
        grads = tape.gradient(loss, self.params)
        self.optimizer.apply_gradients(zip(self.params, grads))
        return loss, accuracy, self.optimizer


if __name__ == '__main__':
    args = parse_args()
    dg.logging.info('Called with args:\n' + str(args))

    np.random.seed(args.seed)
    dg.autograph.set_execution(args.execution)
    dg.cuda.set_default_device(args.cuda)

    # Define the model
    model = Classifier(dg.optimizers.SGD(base_lr=0.01, momentum=0.9, weight_decay=1e-4))

    # Compile for graph execution if necessary
    if args.execution == 'GRAPH_MODE':
        model.step = dg.function(
            func=model.step,
            input_signature=[model.image_spec, model.label_spec],
        )

    # Main loop
    import tensorflow as tf
    batch_size = 200
    for i in range(50):
        for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            image = dg.EagerTensor(X_batch, copy=False)
            label = dg.EagerTensor(y_batch, copy=False, dtype='float32')
            loss, accuracy, _ = model.step(image, label)
            if i % 20 == 0:
                dg.logging.info(
                    'Iteration %d, lr = %s, loss = %.5f, accuracy = %.3f' %
                    (i, str(model.optimizer.base_lr), loss, accuracy)
                )
