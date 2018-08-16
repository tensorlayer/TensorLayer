#! /usr/bin/python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import tensorflow as tf
import tensorlayer as tl

from tensorlayer import logging

from tensorlayer.networks import BaseNetwork

from tensorlayer.decorators import protected_method
from tensorlayer.decorators import private_method

__all__ = ['CustomModel']


class CustomModel(BaseNetwork, ABC):

    def __init__(self, name):
        super(CustomModel, self).__init__(name)

        self.input_layer, self.output_layer = self._check_model_implementation(self.model())

    @abstractmethod
    def model(self):
        raise NotImplementedError("this function must be overwritten")

    @private_method
    def _check_model_implementation(self, model):

        if len(model) != 2:
            raise RuntimeError("The method: CustomModel.model() should return two layers: input_layer and output_layer")
        else:
            input_layer, output_layer = model

        if not isinstance(input_layer,
                          (tl.layers.InputLayer, tl.layers.OneHotInputLayer, tl.layers.Word2vecEmbeddingInputlayer,
                           tl.layers.EmbeddingInputlayer, tl.layers.AverageEmbeddingInputlayer)):
            raise RuntimeError(
                "The returned input layer (type: %s) is not an instance of a known input layer: %s" %
                (type(input_layer), tl.layers.inputs.__all__)
            )

        if not isinstance(output_layer, tl.layers.Layer):
            raise RuntimeError(
                "The returned output layer (type: %s) is not an instance of a `tensorlayer.layers.Layer`" %
                type(output_layer)
            )

        return input_layer, output_layer

    def register_new_layer(self, layer):

        if not isinstance(layer, tl.layers.Layer):
            raise TypeError('You can only register a `tensorlayer.layers.Layer`. Found: %s' % type(layer))

        if layer.name in self.all_layers_dict.keys():
            raise ValueError("The layer name `%s` already exists in this network" % layer.name)

        self.all_layers_dict[layer.name] = layer
        self.all_layers.append(layer.name)

        # Reset Network State in case it was previously compiled
        self._net = None
        self.outputs = None
        self.is_compiled = False

    def compile(self, input_plh, reuse=False, is_train=True):

        logging.info(
            "** Compiling %s `%s` - reuse: %s, is_train: %s **" % (self.__class__.__name__, self.name, reuse, is_train)
        )

        # Reset All Layers' Inputs
        for name, layer in self.all_layers_dict.items():
            layer.inputs = None
            layer.outputs = None

        with logging.temp_handler("    [*]"):

            _net = self.all_layers_dict[self.all_layers[0]](input_plh)

            with tf.variable_scope(self.name, reuse=reuse):
                for layer in self.all_layers[1:]:
                    _net = self.all_layers_dict[layer](prev_layer=_net, is_train=is_train)
                    self.all_drop.update(_net._local_drop)

            if not self.is_compiled:
                self._net = _net
                self.outputs = self._net.outputs
                self.is_compiled = True

        return self.outputs

    def count_layers(self):
        return len(self.all_layers_dict)

    def __getitem__(self, layer_name):
        return self.all_layers_dict[layer_name]
