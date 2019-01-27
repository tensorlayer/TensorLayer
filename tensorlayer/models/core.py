
import numpy as np
from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorlayer.layers import Layer, ModelLayer
from tensorlayer import logging

__all__ = [
    'Model',
]

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

        # Model inputs and outputs
        self._inputs = inputs
        self._outputs = outputs

        # Model converted into a Layer
        self._model_layer = None

        if inputs is None and outputs is None:
            pass

        else:
            # check type of inputs and outputs
            check_order = ['inputs', 'outputs']
            for co, check_argu in enumerate([inputs, outputs]):
                if isinstance(check_argu, Layer):
                    pass
                elif isinstance(check_argu, list):
                    if len(check_argu) == 0:
                        raise ValueError(
                            "The argument `%s` is detected as an empty list. " % check_order[co] +
                            "It should be either Layer or a list of Layer."
                        )
                    for idx in range(len(check_argu)):
                        if not isinstance(check_argu[idx], Layer):
                            raise TypeError(
                                "The argument `%s` should be either Layer or a list of Layer "
                                % (check_order[co]) +
                                "but the %s[%d] is detected as %s"
                                % (check_order[co], idx, type(check_argu[idx]))
                            )
                else:
                    raise TypeError("The argument `%s` should be either Layer or a list of Layer but received %s" %
                                    (check_order[co], type(check_argu)))

            # automatically connecting layers
            outputs_list = self._outputs if isinstance(self._outputs, list) else [self._outputs]
            self._stacked_layers = list()

            for out in outputs_list:
                stacked_layers = list()
                current = out
                while current is not None:
                    stacked_layers.append(current)
                    # FIXME: assume each layer has only one prev layer
                    current = current._input_layer

                if isinstance(self._inputs, list):
                    # check if the input_layer is in self._inputs
                    idx_of_input = self._find_idx_of_inputs(stacked_layers[-1])
                    flag_input_not_found = True if idx_of_input == -1 else False
                else:
                    flag_input_not_found = True if self._inputs is not stacked_layers[-1] else False
                if flag_input_not_found:
                    raise ValueError(
                        "The layer named `%s` not found in the inputs of the model. " % stacked_layers[-1].name +
                        "Please check the argument `inputs` when the model is created."
                    )

                self._stacked_layers.append(stacked_layers)


    def __call__(self, inputs, is_train=None, **kwargs):
        """

        :param inputs: Tensor or list of Tensor, numpy.ndarray of list of numpy.ndarray (if in eager mode)
        :param is_train: boolean
        :return:
        """

        self._check_mode(is_train)

        # set training / inference mode if necessary
        if is_train is not None:
            self._set_mode_for_layers(is_train)

        # if self._input is a list, then it must be a static network
        if isinstance(self._inputs, list):
            if not isinstance(inputs, list):
                raise ValueError("The argument `inputs` should be a list of values but detected as %s." % type(inputs))
            elif len(inputs) != len(self._inputs):
                raise ValueError("The argument `inputs` should be a list with len=%d but detected as len=%d."
                                 % (len(self._inputs), len(inputs)))

        # convert inputs to tensor if it is originally not
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                inputs[idx] = tf.convert_to_tensor(inputs[idx])
        else:
            inputs = tf.convert_to_tensor(inputs)

        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, *inputs):
        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is None:
            raise ValueError("Outputs not defined. Please define inputs and outputs when the model is created. Or overwrite forward() function.")

        results = list()
        memory = dict()

        for stacked_layers in self._stacked_layers:

            # idx_of_input should not be -1 as it has been checked in __init__
            if isinstance(self._inputs, list):
                idx_of_input = self._find_idx_of_inputs(stacked_layers[-1])
                z = inputs[0][idx_of_input]
            else:
                z = inputs[0]

            for layer in stacked_layers[::-1]:
                if layer.name in memory:
                    z = memory[layer.name]
                else:
                    # FIXME: assume each layer has only one prev layer
                    z = layer(z)
                    memory[layer.name] = z
            results.append(z)

        if not isinstance(self._outputs, list):
            return results[0]
        else:
            return results

    @property
    def weights(self):
        if self._weights is not None and len(self._weights) > 0:
            # self._weights already extracted, so do nothing
            pass
        # FIXME: currently using self._outputs to judge static network or dynamic network
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
        if self.is_train != True:
            self.is_train = True
            self._set_mode_for_layers(True)

    def eval(self):
        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    def test(self):
        self.eval()

    def infer(self):
        self.eval()

    def as_layer(self):

        if self._outputs is None:
            raise AttributeError(
                "Dynamic network cannot be converted to Layer."
            )

        if self._model_layer is None:
            self._model_layer = ModelLayer(self)

        return self._model_layer

    def _check_mode(self, is_train):
        # contradiction test
        if is_train is None and self.is_train is None:
            raise ValueError("Training / inference mode not defined. Argument `is_train` should be set as True / False. Otherwise please use `Model.train()` / `Model.eval()` to switch the mode.")
        elif is_train is not None and self.is_train is not None:
            if is_train == self.is_train:
                logging.warning("Training / inference mode redefined redundantly. Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode.")
            else:
                raise AttributeError("Training / inference mode mismatch. The argument `is_train` is set as %s, " % is_train +
                                     "but the mode is currently set as %s. " % ('Training by Model.train()' if self.is_train else 'Inference by Model.eval()') +
                                     "Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode.")

    def _set_mode_for_layers(self, is_train):
        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is not None:
            for stacked_layers in self._stacked_layers:
                for layer in stacked_layers:
                    layer.is_train = is_train
                    # TODO: test THIS
                    if isinstance(layer, ModelLayer):
                        layer.model._set_mode_for_layers(is_train)
        else:
            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("weights")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), ModelLayer):
                        # FIXME: dynamic network cannot be converted to Layer, so this condition never triggered
                        getattr(self, attr).is_train = is_train
                        getattr(self, attr).model._set_mode_for_layers(is_train)
                    elif isinstance(getattr(self, attr), Layer):
                        getattr(self, attr).is_train = is_train
                except Exception:
                    pass

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
