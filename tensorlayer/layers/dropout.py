#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = [
    'Dropout',
]


class Dropout(Module):
    """
    The :class:`Dropout` class is a noise layer which randomly set some
    activations to zero according to a keeping probability.

    Parameters
    ----------
    keep : float
        The keeping probability.
        The lower the probability it is, the more activations are set to zero.
    seed : int or None
        The seed for random dropout.
    name : None or str
        A unique layer name.

    """

    def __init__(self, keep, seed=0, name=None):  #"dropout"):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.seed = seed

        self.build()
        self._built = True

        logging.info("Dropout %s: keep: %f " % (self.name, self.keep))

    def __repr__(self):
        s = ('{classname}(keep={keep}')
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        self.dropout = tl.ops.Dropout(keep=self.keep, seed=self.seed)

    # @tf.function
    def forward(self, inputs):
        if self.is_train:
            outputs = self.dropout(inputs)
        else:
            outputs = inputs
        return outputs


if __name__ == '__main__':
    shapes_do = (20, 16, 50)
    from tensorlayer.layers.inputs import Input
    # from mindspore import context
    # context.set_context(mode=context.GRAPH_MODE)
    inputs_do = Input(shapes_do)
    dropout = Dropout(keep=0.1)(inputs_do)
    print(dropout)
