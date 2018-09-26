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

        if isinstance(input_layer, (list, tuple)):
            for layer in input_layer:
                if layer.__class__.__name__ not in tl.layers.inputs.__all__:
                    raise RuntimeError(
                        "The returned input_layer (type: %s) contains a layer (type: %s) which is not "
                        "a known input layer: %s"
                        % (type(input_layer), type(layer), tl.layers.inputs.__all__)
                    )

        elif isinstance(input_layer, tl.layers.Layer):
            if input_layer.__class__.__name__ not in tl.layers.inputs.__all__:
                raise RuntimeError(
                    "The returned input_layer (type: %s) is not an instance of a known input layer: %s" %
                    (type(input_layer), tl.layers.inputs.__all__)
                )
        else:
            raise RuntimeError(
                "The returned input_layer (type: %s) is not an instance of Layer type or Tuple/List of Layers" %
                (type(input_layer))
            )

        if isinstance(output_layer, (list, tuple)):
            for layer in output_layer:
                if not isinstance(layer, tl.layers.Layer):
                    raise RuntimeError(
                        "The returned output_layer (type: %s) contains a layer (type: %s) which is not "
                        "an instance of type `tl.layers.Layer`" %
                        (type(output_layer), type(layer))
                    )

        elif not isinstance(output_layer, tl.layers.Layer):
            raise RuntimeError(
                "The returned output_layer (type: %s) is not an instance of  type "
                "`tl.layers.Layer` or Tuple/List of `tl.layers.Layer`" %
                (type(input_layer))
            )

        return input_layer, output_layer

    def __getitem__(self, layer_name):
        return self.all_layers_dict[layer_name]
