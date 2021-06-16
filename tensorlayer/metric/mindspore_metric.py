#! /usr/bin/python
# -*- coding: utf-8 -*-

import mindspore.nn as nn
from mindspore.nn.metrics.metric import Metric
__all__ = [
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
]


class Accuracy(object):

    def __init__(self, topk=1):

        self.accuracy = nn.TopKCategoricalAccuracy(k=topk)

    def update(self, y_pred, y_true):

        self.accuracy.update(y_pred, y_true)

    def result(self):

        return self.accuracy.eval()

    def reset(self):

        self.accuracy.clear()


class Auc(object):

    def __init__(self):

        pass

    def update(self, y_pred, y_true):

        raise Exception('Auc metric function not implemented')

    def result(self):

        pass

    def reset(self):

        pass


class Precision(object):

    def __init__(self):

        self.precision = nn.Precision(eval_type="classification")

    def update(self, y_pred, y_true):

        self.precision.update(y_pred, y_true)

    def result(self):

        return self.precision.eval()

    def reset(self):

        self.precision.clear()


class Recall(object):

    def __init__(self):

        self.recall = nn.Recall(eval_type="classification")

    def update(self, y_pred, y_true):

        self.recall.update(y_pred, y_true)

    def result(self):

        return self.recall.eval()

    def reset(self):

        self.recall.clear()
