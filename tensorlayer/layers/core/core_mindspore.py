#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import str2act, _save_weights, _load_weights
from mindspore.nn import Cell
import tensorlayer as tl
from tensorlayer.layers.utils import (get_variable_with_initializer)
from collections import OrderedDict

__all__ = ['Module', 'SequentialLayer', 'LayerList']

_global_layer_name_dict = {}  # TODO: better implementation?


class Module(Cell):

    def __init__(self, name=None, act=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global _global_layer_name_dict
        if name is None:
            prefix = self.__class__.__name__.lower()

            if _global_layer_name_dict.get(prefix) is not None:
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
            else:
                _global_layer_name_dict[prefix] = 0
                name = prefix
            while True:
                if _global_layer_name_dict.get(name) is None:
                    break
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
        else:
            if _global_layer_name_dict.get(name) is not None:
                pass
            else:
                _global_layer_name_dict[name] = 0

        self.name = name

        if isinstance(act, str):
            str_act = str2act(act)

        if act:
            if isinstance(act, str) and (len(act) > 5 and act[0:5] == "lrelu" or len(act) > 10 and act[0:10] == "leaky_relu"):
                self.act = str_act
            elif isinstance(act, str):
                self.act = str_act()
            else:
                self.act = act()
        else:
            self.act = act

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes = []
        self._nodes_fixed = False

        # Layer weight state
        self._all_weights = []
        self._trainable_weights = []
        self._nontrainable_weights = []

        # Layer training state
        self.is_train = True


        # layer forward  state
        self._forward_state = False

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def construct(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def _get_weights(self, var_name, shape, init=tl.initializers.random_normal(), trainable=True):
        """ Get trainable variables. """
        weight = get_variable_with_initializer(
            scope_name=self.name, var_name=var_name, shape=shape, init=init, trainable=trainable
        )
        self.trainable = trainable
        return weight

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format."""
        _save_weights(self, file_path, format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights()."""
        _load_weights(self, file_path, format, in_order, skip)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tl.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tl.get_tensor_shape(tensors)
        return shape_mem

    def _add_node(self, input_tensors, output_tensors):
        """Add a LayerNode for this layer given input_tensors, output_tensors.

        WARINING: This function should not be called from outside, it should only be called
        in layer.__call__ when building static model.

        Parameters
        ----------
        input_tensors : Tensor or a list of tensors
            Input tensors to this layer.
        output_tensors : Tensor or a list of tensors
            Output tensors to this layer.

        """
        raise NotImplementedError

    def set_train(self):
        """
        Sets the cell to training mode.

        The cell itself and all children cells will be set to training mode.

        Args:
            mode (bool): Specifies whether the model is training. Default: True.
        """
        self._phase = 'train'
        self.add_flags_recursive(training=True)
        return self

    def set_eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.models.vgg16()
        >>> net.eval()
        # do evaluation

        """
        self._phase = 'predict'
        self.add_flags_recursive(training=False)
        return self

    def test(self):
        """Set this network in evaluation mode."""
        self.eval()

    def infer(self):
        """Set this network in evaluation mode."""
        self.eval()

    @property
    def trainable_weights(self):
        """
        Returns all trainable weights.

        Returns a list of all trainable parmeters.

        Args:
            recurse (bool): Whether contains the trainable weights of sublayers. Default: True.

        Returns:
            List, the list of trainable weights.
        """
        self._trainable_weights = list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True)))
        return self._trainable_weights

    @property
    def nontrainable_weights(self):
        """
        Returns all untrainable weights.

        Returns a list of all untrainable weights.

        Args:
            recurse (bool): Whether contains the untrainable weights of sublayers. Default: True.

        Returns:
            List, the list of untrainable weights.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))

    @property
    def all_weights(self):
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True))) \
               + list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))


class SequentialLayer(Module):
    """
    Sequential layer container.

    A list of Layers will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of layers can also be passed in.

    Args:
        args (list, OrderedDict): List of subclass of Module.

    Raises:
        TypeError: If the type of the argument is not list or OrderedDict.

    Inputs:
        - **input** (Tensor) - Tensor with shape according to the first Module in the sequence.

    Outputs:
        Tensor, the output Tensor with shape depending on the input and defined sequence of Layers.

    Examples:
        >>> conv = tl.layers.Conv2d(3, 2, 3, pad_mode='valid')
        >>> bn = tl.layers.BatchNorm2d(2)
        >>> relu = tl.ReLU()
        >>> seq = tl.layers.SequentialLayer([conv, bn, relu])
        >>>
        >>> x = tl.layers.Input((1, 3, 4, 4))
        >>> seq(x)
    """
    def __init__(self, *args):
        super(SequentialLayer, self).__init__()
        # self._built = True
        if len(args) == 1:
            layers = args[0]
            if isinstance(layers, list):
                for index, layer in enumerate(layers):
                    self.insert_child_to_layer(str(index), layer)
            elif isinstance(layers, OrderedDict):
                for name, layer in layers.items():
                    self.insert_child_to_layer(name, layer)
            else:
                raise TypeError('Layers must be list or orderedDict')
        else:
            for index, layer in enumerate(args):
                self.insert_child_to_layer(str(index), layer)
        self.layer_list = list(self._layers.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(
                OrderedDict(list(self._layers.items())[index]))
        index = self._valid_index(len(self), index)
        return list(self._layers.values())[index]

    def __setitem__(self, index, layer):
        if self._valid_module(layer):
            index = self._valid_index(len(self), index)
            key = list(self._layers.keys())[index]
            self._layers[key] = layer
            self.layer_list = list(self._layers.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = self._valid_index(len(self), index)
            key = list(self._layers.keys())[index]
            del self._layers[key]
        elif isinstance(index, slice):
            keys = list(self._layers.keys())[index]
            for key in keys:
                del self._layers[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.layer_list = list(self._layers.values())

    def __len__(self):
        return len(self._layers)

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for layer in self._layers.values():
            layer.set_grad(flag)

    def append(self, layer):
        if self._valid_module(layer):
            self._layers[str(len(self))] = layer
        self.layer_list = list(self._layers.values())
        return self

    def build(self, inputs_shape):
        pass

    def forward(self, input_data):
        for layer in self.layer_list:
            input_data = layer(input_data)
        return input_data

    def _valid_index(self, layer_num, index):
        if not isinstance(index, int):
            raise TypeError("Index {} is not int type")
        if not -layer_num <= index < layer_num:
            raise IndexError("Index should be a number in range [{}, {}), but got {}"
                             .format(-layer_num, layer_num, index))
        return index % layer_num

    def _valid_module(self, layer):
        if issubclass(layer.__class__, Module):
            return True
        raise TypeError('Module {} is not subclass of Module'.format(layer))


class LayerList(Module):
    """
    The class :class:`LayerList` is a linear stack of layers.

    The :class:`LayerList` can be created by passing a list of layer instances.
    The given layer instances will be automatically connected one by one.

    Parameters
    ----------
    layers: list of Layer
        A list of layers.
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.

    Methods
    ---------
    __init__()
        Initializing the LayerList.
    weights()
        A collection of weights of all the layer instances.
    build()
        Build the LayerList. The layer instances will be connected automatically one by one.
    forward()
        Forward the computation. The computation will go through all layer instances.
    """

    def __init__(self, layers, name=None):
        """
        Initializing the LayerList given a list of Layer.

        :param layers: list of Layer
        :param name: str or None
        """

        super(LayerList, self).__init__(name=name)
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        pass

    def forward(self, inputs):
        pass

