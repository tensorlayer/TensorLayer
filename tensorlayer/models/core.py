
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

    def __init__(self, inputs, outputs, name):
        '''

        :param inputs: Layer or list of Layer
        :param outputs: Layer or list of Layer
        :param name: str
        '''
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
        self._inputs = inputs if isinstance(inputs, list) else [inputs]
        self._outputs = outputs

        # Model state: train or test
        # self.is_train = is_train

    def __call__(self, inputs, is_train):
        """

        :param inputs: Tensor or list of Tensor, numpy.ndarray (if in eager mode)
        :param is_train: boolean
        :return:
        """

        # convert inputs to tensor if it is originally not
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                if isinstance(inputs[idx], np.ndarray):
                    inputs[idx] = tf.convert_to_tensor(inputs[idx])
        elif isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs)

        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        outputs_list = self._outputs if isinstance(self._outputs, list) else [self._outputs]
        results = list()
        memory = dict()

        for out in outputs_list:
            stacked_layers = list()
            current = out
            while current is not None:
                stacked_layers.append(current)
                # FIXME: assume only one input layer
                current = current._input_layer

            idx_of_input = self._find_idx_of_inputs(stacked_layers[-1])
            z = inputs_list[idx_of_input]

            for layer in stacked_layers[::-1]:
                if layer.name in memory:
                    z = memory[layer.name]
                else:
                    z = layer.forward(z, is_train)
                    memory[layer.name] = z
            results.append(z)

        if not isinstance(self._outputs, list):
            return results[0]
        else:
            return results

    def _find_idx_of_inputs(self, target_input):
        """
        Return the index of the target_input in self._inputs.
        Return -1 if not found.

        :param target_input: the input layer needs to be located
        :return:
        """
        if isinstance(self._inputs, list):
            for idx, input in enumerate(self._inputs):
                if input == target_input:
                    return idx
        return -1

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
