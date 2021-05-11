#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Admax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']

# Add module aliases

# learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'
Adadelta = None

# learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad'
Adagrad = None

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'
Adam = None

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'
Admax = None

# learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
# l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl',l2_shrinkage_regularization_strength=0.0
Ftrl = None

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam',
Nadam = None

# learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop'
RMSprop = None

# learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'
SGD = None

# learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False
Momentum = None


def Lamb(**kwargs):
    raise Exception('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise Exception('LARS optimizer function not implemented')
