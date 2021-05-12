#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf

__all__ = ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop', 'SGD', 'Momentum', 'Lamb', 'LARS']

# Add module aliases

# learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'
Adadelta = tf.optimizers.Adadelta

# learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,name='Adagrad'
Adagrad = tf.optimizers.Adagrad

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'
Adam = tf.optimizers.Adam

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax'
Adamax = tf.optimizers.Adamax

# learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
# l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl',l2_shrinkage_regularization_strength=0.0
Ftrl = tf.optimizers.Ftrl

# learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam',
Nadam = tf.optimizers.Nadam

# learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,name='RMSprop'
RMSprop = tf.optimizers.RMSprop

# learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD'
SGD = tf.optimizers.SGD

# learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False
Momentum = tf.compat.v1.train.MomentumOptimizer


def Lamb(**kwargs):
    raise Exception('Lamb optimizer function not implemented')


def LARS(**kwargs):
    raise Exception('LARS optimizer function not implemented')
