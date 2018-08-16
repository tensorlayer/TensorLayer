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

    def count_layers(self):
        return len(self.all_layers_dict)

    def __getitem__(self, layer_name):
        return self.all_layers_dict[layer_name]
