#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def generator(X_train, y_train=None):
    inputs = X_train
    targets = y_train
    if targets is None:
        for _input in X_train:
            yield _input
    else:
        if len(inputs) != len(targets):
            raise AssertionError("The length of inputs and targets should be equal")
        for _input, _target in zip(inputs, targets):
            # yield _input.encode('utf-8'), _target.encode('utf-8')
            yield (_input, np.array([_target]))
