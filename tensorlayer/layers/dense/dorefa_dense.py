#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig

from tensorlayer.layers.utils import cabs
from tensorlayer.layers.utils import quantize_active
from tensorlayer.layers.utils import quantize_weight

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

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
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
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
            W_init=tf.compat.v1.initializers.truncated_normal(stddev=0.1),
            b_init=tf.compat.v1.initializers.constant(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name=None,  #'dorefa_dense',
    ):
        # super(DorefaDense, self
        #      ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        super().__init__(name)
        self.bitW = bitW
        self.bitA = bitA
        self.n_units = n_units
        self.act = act
        self.use_gemm = use_gemm
        self.W_init = W_init
        self.b_init = b_init
        self.W_init_args = W_init_args
        self.b_init_args = b_init_args
        logging.info(
            "DorefaDense  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):
        if len(inputs_shape) != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")
        if self.use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = inputs_shape[-1]
        self.W = self._get_weight("weights", shape=(n_in, self.n_units), init=self.W_init, init_args=self.W_init_args)
        # self.W = tf.compat.v1.get_variable(
        #     name=self.name + '\W', shape=(n_in, self.n_units), initializer=self.W_init, dtype=LayersConfig.tf_dtype,
        #     **self.W_init_args
        # )
        if self.b_init is not None:
            self.b = self._get_weight("biases", shape=(self.n_units), init=self.b_init, init_args=self.b_init_args)
        #     try:
        #         self.b = tf.compat.v1.get_variable(
        #             name=self.name + '\b', shape=(self.n_units), initializer=self.b_init, dtype=LayersConfig.tf_dtype,
        #             **self.b_init_args
        #         )
        #
        #     except Exception:  # If initializer is a constant, do not specify shape.
        #         self.b = tf.compat.v1.get_variable(
        #             name=self.name + '\b', initializer=self.b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
        #         )
        #     self.get_weights([self.W, self.b])
        # else:
        #     self.get_weights(self.W)

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
