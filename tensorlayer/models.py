#! /usr/bin/python
# -*- coding: utf-8 -*-

"""A file containing various activation functions."""

__all__ = ["CompiledNetwork"]


class CompiledNetwork(object):

    def __init__(self, inputs, outputs, all_layers_dict, all_params, all_graphs, all_drop, is_train):

        self.inputs = inputs
        self.outputs = outputs

        self.all_params = all_params
        self.all_graphs = all_graphs

        self.all_layers_dict = all_layers_dict

        self.all_drop = all_drop

        self.is_train = is_train

    def __getattribute__(self, item):
        if item == "all_layers":
            return list(self.all_layers_dict.values())
        else:
            return super(CompiledNetwork, self).__getattribute__(item)

    def __getitem__(self, layer_name):
        if layer_name in self.all_layers_dict.keys():
            return self.all_layers_dict[layer_name]

        elif self.model_scope + "/" + layer_name in self.all_layers_dict.keys():
            return self.all_layers_dict[self.model_scope + "/" + layer_name]

        else:
            raise ValueError("layer name `%s` does not exist in this network" % layer_name)

    def __setattr__(self, key, value):
        if not hasattr(self, key):
            super(CompiledNetwork, self).__setattr__(key, value)
        else:
            raise RuntimeError(
                "A Tensorlayer network is not supposed to be modified once it has been compiled. "
                "An attempt to modify the attribute: `{}` has been detected.".format(key)
            )

    def count_params(self):
        """Returns the number of parameters in the network."""
        n_params = 0

        for _i, p in enumerate(self.all_params):

            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():

                try:
                    s = int(s)
                except TypeError:
                    s = 1

                if s:
                    n = n * s

            n_params = n_params + n

        return n_params
