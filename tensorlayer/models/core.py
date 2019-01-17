
import numpy as np
import tensorflow as tf
from tensorlayer.layers import Layer

class Model():
    """The :class:`Model` class represents a neural network.

    Parameters
    -----------
    inputs : a Layer or list of Layer
        The input(s) to the model.
    outputs : a Layer or list of Layer
        The output(s) to the model.
    name : None or str
        The name of the model.
    """

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs, outputs, name=None):
        # Model properties
        self.name = name

        # check type of inputs and outputs
        check_order = ['inputs', 'outputs']
        for co, check_argu in enumerate([inputs, outputs]):
            if isinstance(check_argu, Layer):
                pass
            elif isinstance(check_argu, list):
                for idx in range(len(check_argu)):
                    if not isinstance(check_argu[idx], Layer):
                        raise TypeError(
                            "The argument %s should be either Layer or a list of Layer "
                            % (check_order[co]) +
                            "but the %s[%d] is detected as %s"
                            % (check_order[co], idx, type(check_argu[idx]))
                        )
            else:
                raise TypeError("The argument %s should be either Layer or a list of Layer but received %s" %
                                (check_order[co], type(check_argu)))

        # Model inputs and outputs
        self._inputs = inputs
        self._outputs = outputs

        # Model state: train or test
        # self.is_train = is_train

    def __call__(self, inputs, is_train):

        # convert inputs to tensor if it is originally not
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                if isinstance(inputs[idx], np.ndarray):
                    inputs[idx] = tf.convert_to_tensor(inputs[idx])
        elif isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs)

        # check inputs
        if isinstance(self._inputs, Layer):
            print(self._inputs._outputs_shape)
            print(inputs.get_shape().as_list())
        exit()

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

    def __str__(self):
        return "  {} ({}) outputs_shape: {}".format(
            self.__class__.__name__, self.name, [tuple(['batch_size'] + o._outputs_shape[1:]) for o in self.outputs]
        )  #_outputs_shape)#outputs.get_shape().as_list())

    def print_all_layers(self):
        for out in self._outputs:
            stacked_layers = list()
            current = out
            while current is not None:
                print(current.name, current == self._inputs)
                stacked_layers.append(current)
                current = current._input_layer
        pass

    ## raise Exceptions for old version codes
    def count_params(self, **kwargs):
        raise Exception("please change count_params --> count_weights")

    def print_params(self, **kwargs):
        raise Exception("please change print_params --> print_weights")

    @property
    def all_params(self):
        raise Exception("please change all_params --> weights")

    @property
    def all_drop(self):
        raise Exception("all_drop is deprecated")
