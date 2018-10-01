#! /usr/bin/python
# -*- coding: utf-8 -*-

# """A file containing various activation functions."""

import tensorflow as tf
import tensorlayer as tl

from tensorlayer.decorators import deprecated

__all__ = ["BuiltNetwork"]


class BuiltNetwork(object):

    def __init__(self, inputs, outputs, all_layers, is_train, model_scope, name):

        self.inputs = inputs
        self.outputs = outputs

        self.all_layers = {layer.name: layer for layer in all_layers}

        self.all_drop = dict()
        self.all_weights = list()

        for layer in self.all_layers.values():
            self.all_drop.update(layer.local_drop)
            self.all_weights.extend(layer.local_weights)

        self.is_train = is_train

        self.model_scope = model_scope
        self.name = name

    def __getattribute__(self, item):

        if item == "all_params":
            tl.logging.warning(
                "`all_params` has been deprecated in favor of `all_weights and will be removed in a future version`"
            )
            return self.all_weights

        else:
            return super(BuiltNetwork, self).__getattribute__(item)

    def __getitem__(self, layer_name):
        if layer_name in self.all_layers.keys():
            return self.all_layers[layer_name]

        elif self.model_scope + "/" + layer_name in self.all_layers.keys():
            return self.all_layers[self.model_scope + "/" + layer_name]

        else:
            raise ValueError("layer name `%s` does not exist in this network" % layer_name)

    def __setattr__(self, key, value):
        if not hasattr(self, key):
            super(BuiltNetwork, self).__setattr__(key, value)
        else:
            raise RuntimeError(
                "A Tensorlayer `{}` is not supposed to be modified. "
                "An attempt to modify the attribute: `{}` has been detected.".format(self.__class__.__name__, key)
            )

    # =============================================== #
    #                 PRIVATE METHODS                 #
    # =============================================== #

    # =============================================== #
    #                PROTECTED METHODS                #
    # =============================================== #

    # =============================================== #
    #                  PUBLIC METHODS                 #
    # =============================================== #

    def count_layers(self):
        return len(self.all_layers)

    def count_weights(self):
        """Returns the number of parameters in the network."""
        n_params = 0

        for _i, p in enumerate(self.all_weights):

            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():

                try:
                    s = int(s)
                except (TypeError, ValueError):
                    s = 1

                if s:
                    n = n * s

            n_params = n_params + n

        return n_params

    def get_all_weights(self):
        """Returns a list of parameters in the network"""
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope)

    # =============================================== #
    #              TO BE REMOVED/MOVED                #
    # =============================================== #

    @deprecated(
        end_support_version="2.1.0", instructions="`count_params` has been deprecated in favor of `count_weights`"
    )  # TODO: remove this line before releasing TL 2.1.0
    def count_params(self):
        """Returns the number of parameters in the network"""
        return self.count_weights()

    @deprecated(
        end_support_version="2.1.0", instructions="`get_all_params` has been deprecated in favor of `get_all_weights`"
    )  # TODO: remove this line before releasing TL 2.1.0
    def get_all_params(self):
        return self.get_all_weights()
