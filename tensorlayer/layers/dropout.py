#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Dropout',
]


class Dropout(Layer):

    def __init__(self, keep, seed, name=None):  #"dropout"):
        super().__init__(name)
        self.keep = keep
        self.seed = seed
        self.name = name
        logging.info("Dropout %s: keep: %f " % (self.name, keep))

    def build(self, inputs):
        pass

    def forward(self, inputs, is_train):
        if is_train:
            outputs = tf.nn.dropout(inputs, keep=self.keep, seed=self.seed, name=self.name)
        else:
            outputs = inputs
        return outputs
