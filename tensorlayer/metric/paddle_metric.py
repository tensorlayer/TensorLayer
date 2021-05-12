#! /usr/bin/python
# -*- coding: utf-8 -*-

import paddle
from paddle.metric.metrics import Metric

__all__ = [
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
]


class Accuracy(object):

    def __init__(
        self,
        topk=1,
    ):

        self.topk = topk
        self.accuracy = paddle.metric.Accuracy(topk=(self.topk, ))

    def update(self, y_pred, y_true):

        self.accuracy.update(self.accuracy.compute(y_pred, y_true))

    def result(self):

        return self.accuracy.accumulate()

    def reset(self):

        self.accuracy.reset()


class Auc(object):

    def __init__(self, curve='ROC', num_thresholds=4095):

        self.auc = paddle.metric.Auc(curve=curve, num_thresholds=num_thresholds)

    def update(self, y_pred, y_true):

        self.auc.update(y_pred, y_true)

    def result(self):

        return self.auc.accumulate()

    def reset(self):

        self.auc.reset()


class Precision(object):

    def __init__(self):

        self.precision = paddle.metric.Precision()

    def update(self, y_pred, y_true):

        self.precision.update(y_pred, y_true)

    def result(self):

        return self.precision.accumulate()

    def reset(self):

        self.precision.reset()


class Recall(object):

    def __init__(self):

        self.recall = paddle.metric.Recall()

    def update(self, y_pred, y_true):
        self.recall.update(y_pred, y_true)

    def result(self):
        return self.recall.accumulate()

    def reset(self):
        self.recall.reset()
