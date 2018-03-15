# -*- coding: utf-8 -*-

from .core import *
from .. import _logging as logging
import tensorflow as tf
import tensorlayer as tl

__all__ = [
    'BinaryDenseLayer',
    'SignLayer',
    'MultiplyScaleLayer',
]


@tf.RegisterGradient("TL_Sign_QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

def quantize(x):
    with tf.get_default_graph().gradient_override_map({"Sign": "TL_Sign_QuantizeGrad"}):
        return tf.sign(x)

class BinaryDenseLayer(Layer): # https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    """The :class:`BinaryDenseLayer` class is a binary fully connected layer, which weights are either -1 or 1 while inferencing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`SignLayer` after :class:`BatchNormLayer`.
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
            prev_layer,
            n_units=100,
            act=tf.identity,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            W_init_args=None,
            name='binary_dense',
    ):
        if W_init_args is None:
            W_init_args = {}

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        logging.info("BinaryDenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args)
            # W = tl.act.sign(W)
            W = quantize(W)
            # W = tf.Variable(W)
            print(W)
            self.outputs = act(tf.matmul(self.inputs, W))
            # self.outputs = act(xnor_gemm(self.inputs, W)) # TODO

        self.all_layers.append(self.outputs)
        self.all_params.append(W)

class SignLayer(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            name='sign',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("SignLayer  %s" % (self.name))
        with tf.variable_scope(name):
            # self.outputs = tl.act.sign(self.inputs)
            self.outputs = quantize(self.inputs)

        self.all_layers.append(self.outputs)

class MultiplyScaleLayer(Layer):
    """The :class:`AddScaleLayer` class is for multipling a trainble scale value to the layer outputs. Usually be used on the output of binary net.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer.
    init_scale : float
        The initial value for the scale factor.
    name : a str
        A unique layer name.

    """

    def __init__(
            self,
            prev_layer,
            init_scale=0.05,
            name='scale',
    ):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs

        logging.info("MultiplyScaleLayer  %s: init_scale: %f" % (self.name, init_scale))
        with tf.variable_scope(name):
            # scale = tf.get_variable(name='scale_factor', init, trainable=True, )
            scale = tf.get_variable("scale", shape=[1], initializer=tf.constant_initializer(value=init_scale))
            self.outputs = self.inputs * scale

        self.all_layers.append(self.outputs)
        self.all_params.append(scale)
