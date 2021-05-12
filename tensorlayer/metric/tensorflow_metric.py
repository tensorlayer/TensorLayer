#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.metrics import Metric

__all__ = [
    'Accuracy',
    'Auc',
    'Precision',
    'Recall',
]


class Accuracy(object):

    def __init__(self, topk=1):
        self.topk = topk
        if topk == 1:
            self.accuary = tf.keras.metrics.Accuracy()
        else:
            self.accuary = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=topk)

    def update(self, y_pred, y_true):

        if self.topk == 1:
            y_pred = tf.argmax(y_pred, axis=1)
            self.accuary.update_state(y_true, y_pred)
        else:
            self.accuary.update_state(y_true, y_pred)

    def result(self):

        return self.accuary.result()

    def reset(self):

        self.accuary.reset_states()


class Auc(object):

    def __init__(
        self,
        curve='ROC',
        num_thresholds=200,
    ):
        self.auc = tf.keras.metrics.AUC(num_thresholds=num_thresholds, curve=curve)

    def update(self, y_pred, y_true):

        self.auc.update_state(y_true, y_pred)

    def result(self):

        return self.auc.result()

    def reset(self):

        self.auc.reset_states()


class Precision(object):

    def __init__(self):

        self.precision = tf.keras.metrics.Precision()

    def update(self, y_pred, y_true):

        self.precision.update_state(y_true, y_pred)

    def result(self):

        return self.precision.result()

    def reset(self):

        self.precision.reset_states()


class Recall(object):

    def __init__(self):

        self.recall = tf.keras.metrics.Recall()

    def update(self, y_pred, y_true):

        self.recall.update_state(y_true, y_pred)

    def result(self):

        return self.recall.result()

    def reset(self):

        self.recall.reset_states()
