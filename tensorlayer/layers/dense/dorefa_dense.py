#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.layers.utils import cabs, quantize_active, quantize_weight

__all__ = [
    'DorefaDense',
]


class DorefaDense(Layer):
    """The :class:`DorefaDense` class is a binary fully connected layer, which weights are 'bitW' bits and the output of the previous layer
    are 'bitA' bits while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    bitW : int
        The bits of this layer's parameter
    bitA : int
        The bits of the output of previous layer
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`Sign` after :class:`BatchNorm`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            bitW=1,
            bitA=3,
            n_units=100,
            act=None,
            use_gemm=False,
            W_init=tl.initializers.truncated_normal(stddev=0.1),
            b_init=tl.initializers.constant(value=0.0),
            in_channels=None,
            name=None,  #'dorefa_dense',
    ):
        super().__init__(name)
        self.bitW = bitW
        self.bitA = bitA
        self.n_units = n_units
        self.act = act
        self.use_gemm = use_gemm
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build((None, self.in_channels))
            self._built = True

        logging.info(
            "DorefaDense  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        s += ', bitW={bitW}, bitA={bitA}'
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if self.in_channels is None:
            self.in_channels = inputs_shape[1]

        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = inputs_shape[-1]
        self.W = self._get_weights("weights", shape=(n_in, self.n_units), init=self.W_init)
        if self.b_init is not None:
            self.b = self._get_weights("biases", shape=(self.n_units), init=self.b_init)

    def forward(self, inputs):
        inputs = quantize_active(cabs(inputs), self.bitA)
        W_ = quantize_weight(self.W, self.bitW)
        outputs = tf.matmul(inputs, W_)
        # self.outputs = xnor_gemm(self.inputs, W) # TODO
        if self.b_init is not None:
            outputs = tf.nn.bias_add(outputs, self.b, name='bias_add')
            # self.outputs = xnor_gemm(self.inputs, W) + b # TODO
        if self.act:
            outputs = self.act(outputs)
        return outputs
