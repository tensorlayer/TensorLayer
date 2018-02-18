# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class KerasLayer(Layer):
    """
    The :class:`KerasLayer` class can be used to merge all Keras layers into
    TensorLayer. Example can be found here `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_.
    This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    keras_layer : function
        A tensor in tensor out function for building model.
    keras_args : dictionary
        The arguments for the `keras_layer`.
    name : str
        A unique layer name.
    """

    def __init__(
            self,
            layer,
            keras_layer,
            keras_args={},
            name='keras_layer',
    ):
        Layer.__init__(self, name=name)
        assert layer is not None
        assert keras_layer is not None
        self.inputs = layer.outputs
        logging.info("KerasLayer %s: %s" % (self.name, keras_layer))
        logging.info("This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = keras_layer(self.inputs, **keras_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


class EstimatorLayer(Layer):
    """
    The :class:`EstimatorLayer` class accepts ``model_fn`` that described the model.
    It is similar with :class:`KerasLayer`, see `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`_.
    This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    model_fn : function
        A tensor in tensor out function for building model.
    args : dictionary
        The arguments for the `model_fn`.
    name : str
        A unique layer name.
    """

    def __init__(
            self,
            layer,
            model_fn,
            args={},
            name='estimator_layer',
    ):
        Layer.__init__(self, name=name)
        assert layer is not None
        assert model_fn is not None
        self.inputs = layer.outputs
        logging.info("EstimatorLayer %s: %s" % (self.name, model_fn))
        logging.info("This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = model_fn(self.inputs, **args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
