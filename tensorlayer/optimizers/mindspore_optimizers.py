#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from mindspore.nn import optim as optimizer
import mindspore as ms
from mindspore.nn import Cell

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']


class Adadelta(Cell):

    def __init__(self):
        pass

    def app_gradients(self):
        raise Exception('Adadelta optimizer function not implemented')


class Adagrad(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Adagrad optimizer function not implemented')


class Adam(Cell):

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        self.adam = optimizer.Adam
        self.learn_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_adam = self.adam(
            vars, learning_rate=self.learn_rate, beta1=self.beta_1, beta2=self.beta_2, eps=self.epsilon
        )
        optimizer_adam(grads)


class Adamax(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Adamax optimizer function not implemented')


class Ftrl(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Ftrl optimizer function not implemented')


class Nadam(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('Nadam optimizer function not implemented')


class RMSprop(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('RMSprop optimizer function not implemented')


class RMSprop(Cell):

    def __init__(self):
        pass

    def apply_gradients(self):
        raise Exception('RMSprop optimizer function not implemented')


class SGD(Cell):

    def __init__(self, learning_rate, momentum):
        self.sgd = optimizer.SGD
        self.learn_rate = learning_rate
        self.momentum = momentum

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_sgd = self.sgd(vars, learning_rate=self.learn_rate, momentum=self.momentum)
        optimizer_sgd(grads)


class Momentum(Cell):

    def __init__(self, learning_rate, momentum):
        self.mom = optimizer.Momentum
        self.learn_rate = learning_rate
        self.momentum = momentum

    def apply_gradients(self, grads_and_vars, **kwargs):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_mom = self.mom(vars, learning_rate=self.learn_rate, momentum=self.momentum, **kwargs)
        optimizer_mom(grads)


class Lamb(Cell):

    def __init__(
        self, decay_steps, warmup_steps=0, start_learning_rate=0.1, end_learning_rate=0.0001, power=1.0, beta1=0.9,
        beta2=0.999, eps=1e-06, weight_decay=0.0
    ):
        self.lamb = optimizer.Lamb
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars):
        grads, vars = list(zip(*grads_and_vars))
        optimizer_lamb = self.lamb(
            params=vars, decay_steps=self.decay_steps, warmup_steps=self.warmup_steps,
            start_learning_rate=self.start_learning_rate, end_learning_rate=self.end_learning_rate, power=self.power,
            beta1=self.beta1, beta2=self.beta2, eps=self.eps, weight_decay=self.weight_decay
        )
        optimizer_lamb(grads)


class LARS(object):

    def __init__(self, optimizer, **kwargs):
        self.lars = ms.nn.LARS(optimizer=optimizer, **kwargs)

    def apply_gradients(self, grads_and_vars):
        grads, _ = list(zip(*grads_and_vars))
        self.lars(grads)
