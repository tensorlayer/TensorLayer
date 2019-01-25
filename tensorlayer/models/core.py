
import numpy as np
import tensorflow as tf
from tensorlayer.layers import Layer
from tensorlayer import logging

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

    def __init__(self, inputs=None, outputs=None, name=None):
        '''

        :param inputs: Layer or list of Layer
        :param outputs: Layer or list of Layer
        :param name: str
        '''
        # Model properties
        # TODO: model auto naming
        self.name = name

        # Model state: train or test
        self.is_train = None

        # Model weights
        self._weights = None

        if inputs is None and outputs is None:
            pass

        else:
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


    def __call__(self, inputs, is_train=None, **kwargs):
        """

        :param inputs: Tensor or list of Tensor, numpy.ndarray of list of numpy.ndarray (if in eager mode)
        :param is_train: boolean
        :return:
        """

        if is_train is None and self.is_train is None:
            raise ValueError("Training / inference mode not defined. Argument `is_train` should be set as True / False. Otherwise please use `Model.train()` / `Model.eval()` to switch the mode.")
        elif is_train is not None and self.is_train is not None:
            if is_train == self.is_train:
                logging.warning("Training / inference mode redefined redundantly. Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode.")
            else:
                raise AttributeError("Training / inference mode mismatch. The argument `is_train` is set as %s, " % is_train +
                                     "but the mode is currently set as %s. " % ('Training by Model.train()' if self.is_train else 'Inference by Model.eval()') +
                                     "Please EITHER use the argument `is_train` OR `Model.train()` / `Model.test()` to define the mode.")

        # convert inputs to tensor if it is originally not
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                inputs[idx] = tf.convert_to_tensor(inputs[idx])
        else:
            inputs = tf.convert_to_tensor(inputs)

        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is not None:
            # self._inputs and self._outputs are defined when the model is created

            # convert inputs to list for convenience
            # inputs_list = inputs if isinstance(inputs, list) else [inputs]
            outputs_list = self._outputs if isinstance(self._outputs, list) else [self._outputs]
            results = list()
            memory = dict()

            for out in outputs_list:
                stacked_layers = list()
                current = out
                while current is not None:
                    stacked_layers.append(current)
                    # FIXME: assume each layer has only one prev layer
                    current = current._input_layer

                if isinstance(self.inputs, list):
                    idx_of_input = self._find_idx_of_inputs(stacked_layers[-1])
                    z = inputs[idx_of_input]
                else:
                    z = inputs

                for layer in stacked_layers[::-1]:
                    if layer.name in memory:
                        z = memory[layer.name]
                    else:
                        # FIXME: not sure if there is a better way
                        layer.is_train = is_train if is_train is not None else self.is_train
                        # FIXME: assume each layer has only one prev layer
                        # z = layer.forward(z)
                        z = layer(z)
                        memory[layer.name] = z
                results.append(z)

            if not isinstance(self._outputs, list):
                return results[0]
            else:
                return results
        else:
            # self._inputs and self._outputs are NOT defined when self is created (eager mode)

            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("weights")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), Layer):
                        getattr(self, attr).is_train = is_train if is_train is not None else self.is_train
                except Exception:
                    pass

            return self.forward(inputs, **kwargs)


    @property
    def weights(self):
        if self._weights is not None and len(self._weights) > 0:
            # self._weights already extracted, so do nothing
            pass
        elif self._outputs is not None:
            # self._inputs and self._outputs are defined when self is created
            self._weights = list()
            outputs_list = self._outputs if isinstance(self._outputs, list) else [self._outputs]
            for out in outputs_list:
                current = out
                while current is not None:
                    if current.weights is not None:
                        self._weights.extend(current.weights)
                    # FIXME: assume each layer has only one prev layer
                    current = current._input_layer
        else:
            # self._inputs and self._outputs are NOT defined when self is created (eager mode)
            self._weights = list()
            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("weights")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), Layer):
                        self._weights.extend(getattr(self, attr).weights)
                except Exception:
                    pass

        return self._weights

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def test(self):
        self.eval()

    def infer(self):
        self.eval()

    '''
    def _set_mode_for_layers(self, is_train):
        attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
        attr_list.remove("weights")
        for idx, attr in enumerate(attr_list):
            try:
                if isinstance(getattr(self, attr), Layer):
                    getattr(self, attr).is_train = is_train 
            except Exception:
                pass
    '''


    def _find_idx_of_inputs(self, target_input):
        """
        Return the index of the target_input in self._inputs.
        Return -1 if not found.

        :param target_input: the input layer needs to be located
        :return:
        """
        if isinstance(self._inputs, list):
            for idx, input in enumerate(self._inputs):
                if input is target_input:
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
