

import tensorflow as tf

class Model():

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs=None, outputs=None, name="mymodel"):
        # Model properties
        self.name = name

        # Model inputs and outputs
        # TODO: check type of inputs and outputs
        self._inputs = inputs
        self._outputs = outputs

        # Model state: train or test
        # self.is_train = is_train

    def __call__(self, inputs, is_train):
        # TODO: check inputs corresponds with self._inputs
        results = list()
        for out in self._outputs:
            stacked_layers = list()
            current = out
            # TODO: if inputs is not Input but BaseLayer?
            while current is not None:
                stacked_layers.append(current)
                current = current._input_layer
            # FIXME: assume there is only one inputs
            z = inputs
            for layer in stacked_layers[::-1]:
                z = layer.forward(z, is_train)
            results.append(z)
        return results

