#! /usr/bin/python
# -*- coding: utf-8 -*-

from abc import abstractmethod

import tensorflow as tf

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.decorators import (deprecated_alias, private_method, protected_method)
from tensorlayer.layers.utils import (get_variable_with_initializer, list_remove_repeat)
from tensorlayer.files import utils

import inspect

__all__ = ['Layer', 'ModelLayer', 'LayerList']

_global_layer_name_dict = {}  # TODO: better implementation?


class Layer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.

    Parameters
    ----------
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.

    Methods
    ---------
    __init__()
        Initializing the Layer.
    __call__()
        (1) Building the Layer if necessary. (2) Forwarding the computation.
    all_weights()
        Return a list of Tensor which are all weights of this Layer.
    trainable_weights()
        Return a list of Tensor which are all trainable weights of this Layer.
    nontrainable_weights()
        Return a list of Tensor which are all nontrainable weights of this Layer.
    build()
        Abstract method. Build the Layer. All trainable weights should be defined in this function.
    forward()
        Abstract method. Forward computation and return computation results.

    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Initializing the Layer.

        :param name: str or None
        """

        # Layer constants
        # for key in kwargs.keys():
        #     setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        # Auto naming if the name is not given
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
                # raise ValueError(
                #     'Layer name \'%s\' has already been used by another layer. Please change the layer name.' % name
                # )
            else:
                _global_layer_name_dict[name] = 0

        self.name = name

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes = []
        self._nodes_fixed = False

        # Layer weight state
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # Layer training state
        self.is_train = True

        # layer config and init_args
        self._config = None
        self.layer_args = self._get_init_args(skip=3)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [t.get_shape().as_list() for t in tensors]
        else:
            shape_mem = tensors.get_shape().as_list()
        return shape_mem

    @property
    def config(self):
        # if not self._nodes_fixed:
        #     raise RuntimeError("Model can not be saved when nodes are not fixed.")
        if self._config is not None:
            return self._config
        else:
            _config = {}
            _config.update({'class': self.__class__.__name__.split('.')[-1]})
            self.layer_args.update(self.get_args())
            self.layer_args["name"] = self.name
            _config.update({"args": self.layer_args})
            if self.__class__.__name__ in tl.layers.inputs.__all__:
                _config.update({'prev_layer': None})
            else:
                _config.update({'prev_layer': []})
                for node in self._nodes:
                    in_nodes = node.in_nodes
                    if not isinstance(in_nodes, list):
                        prev_name = in_nodes.name
                    else:
                        prev_name = [in_node.name for in_node in in_nodes]
                        if len(prev_name) == 1:
                            prev_name = prev_name[0]
                    _config['prev_layer'].append(prev_name)
            if self._nodes_fixed:
                self._config = _config
            return _config

    @property
    def all_weights(self):
        if self._all_weights is not None and len(self._all_weights) > 0:
            pass
        else:
            self._all_weights = list()
            if self._trainable_weights is not None:
                self._all_weights.extend(self._trainable_weights)
            if self._nontrainable_weights is not None:
                self._all_weights.extend(self._nontrainable_weights)
        return self._all_weights

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @property
    def nontrainable_weights(self):
        return self._nontrainable_weights

    def __call__(self, inputs, *args, **kwargs):
        """
        (1) Build the Layer if necessary.
        (2) Forward the computation and return results.
        (3) Add LayerNode if necessary

        :param prev_layer: np.ndarray, Tensor, Layer, list of Layers
        :param kwargs:
        :return: Layer
        """
        if self.__class__.__name__ in tl.layers.inputs.__all__:
            input_tensors = tf.convert_to_tensor(inputs)
        else:
            input_tensors = inputs

        if not self._built:
            if isinstance(self, LayerList):
                self._input_tensors = input_tensors
            inputs_shape = self._compute_shape(input_tensors)
            self.build(inputs_shape)
            self._built = True

        outputs = self.forward(input_tensors, *args, **kwargs)

        if not self._nodes_fixed:
            self._add_node(input_tensors, outputs)

        return outputs

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
        inputs_list = tolist(input_tensors)
        outputs_list = tolist(output_tensors)

        if self.__class__.__name__ in tl.layers.inputs.__all__:
            # for InputLayer, there should be no in_nodes
            in_nodes = []
            in_tensor_idxes = [0]
        else:
            in_nodes = [tensor._info[0] for tensor in inputs_list]
            in_tensor_idxes = [tensor._info[1] for tensor in inputs_list]
        node_index = len(self._nodes)

        new_node = LayerNode(self, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes)
        self._nodes.append(new_node)
        for idx, tensor in enumerate(outputs_list):
            tensor._info = (new_node, idx)  # FIXME : modify tensor outside layers? how to deal?

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted in order to release memory.
        """
        # FIXME : not understand why saving inputs/outputs shape
        for node in self._nodes:
            node.in_tensors = None
            node.out_tensors = None

    def _set_mode_for_layers(self, is_train):
        """ Set training/evaluation mode for the Layer"""
        self.is_train = is_train

    def _fix_nodes_for_layers(self):
        """ fix LayerNodes to stop growing for this layer"""
        self._nodes_fixed = True

    def _get_weights(self, var_name, shape, init=tl.initializers.random_normal(), trainable=True):
        """ Get trainable variables. """
        weight = get_variable_with_initializer(scope_name=self.name, var_name=var_name, shape=shape, init=init)
        if trainable is True:
            if self._trainable_weights is None:
                self._trainable_weights = list()
            self._trainable_weights.append(weight)
        else:
            if self._nontrainable_weights is None:
                self._nontrainable_weights = list()
            self._nontrainable_weights.append(weight)
        return weight

    @abstractmethod
    def build(self, inputs_shape):
        """
        An abstract method which should be overwritten in derived classes
        to define all necessary trainable weights of the layer.

        self.built should be set as True after self.build() is called.

        :param inputs_shape: tuple
        """
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    @abstractmethod
    def forward(self, inputs):
        """
        An abstract method which should be overwritten in derived classes
        to define forward feeding operations of the layer.

        :param inputs: Tensor
        :return: Tensor
        """
        raise Exception("The forward method must be implemented by inherited class")

    @abstractmethod
    def __repr__(self):
        reprstr = "Layer"
        return reprstr

    def __setitem__(self, key, item):
        raise TypeError("The Layer API does not allow to use the method: `__setitem__`")

    def __delitem__(self, key):
        raise TypeError("The Layer API does not allow to use the method: `__delitem__`")

    @protected_method
    def get_args(self):
        init_args = {"layer_type": "normal"}
        return init_args

    @protected_method
    def _get_init_args(self, skip=3):
        """Get all arguments of current layer for saving the graph."""
        stack = inspect.stack()

        if len(stack) < skip + 1:
            raise ValueError("The length of the inspection stack is shorter than the requested start position.")

        args, _, _, values = inspect.getargvalues(stack[skip][0])

        params = {}

        for arg in args:

            # some args dont need to be saved into the graph. e.g. the input placeholder
            if values[arg] is not None and arg not in ['self', 'prev_layer', 'inputs']:

                val = values[arg]

                # change function (e.g. act) into dictionary of module path and function name
                if inspect.isfunction(val):
                    params[arg] = ('is_Func', utils.func2str(val))
                    # if val.__name__ == "<lambda>":
                    #     params[arg] = utils.lambda2str(val)
                    # else:
                    #     params[arg] = {"module_path": val.__module__, "func_name": val.__name__}
                # ignore more args e.g. TL initializer
                elif arg.endswith('init'):
                    continue
                # for other data type, save them directly
                else:
                    params[arg] = val

        return params


class LayerNode(object):
    """
    The class :class:`LayerNode` class represents a conceptional node for a layer.

    LayerNode is used for building static model and it is actually a light weighted
    wrapper over Layer. Specifically, it is used for building static computational graph
    (see _construct_graph() in tl.models.Model). In static model, each layer relates to
    one or more LayerNode, and the connection relationship between layers is built upon
    LayerNode. In addition, LayerNode eases layer reuse and weights sharing.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of LayerNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    Methods
    ---------
    __init__()
        Initializing the LayerNode.
    __call__()
        (1) Forwarding through the layer. (2) Update its input/output tensors.
    """

    def __init__(self, layer, node_index, in_nodes, in_tensors, out_tensors, in_tensor_idxes):
        """

        Parameters
        ----------
        layer
        node_index
        in_nodes
        in_tensors
        out_tensors
        in_tensor_idxes
        """
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.name = layer.name + "_node_{}".format(node_index)

        self.in_tensors_idxes = in_tensor_idxes

        self.visited = False

    def __call__(self, inputs, **kwargs):
        """(1) Forwarding through the layer. (2) Update its input/output tensors."""
        outputs = self.layer.forward(inputs, **kwargs)
        self.in_tensors = tolist(inputs)
        self.out_tensors = tolist(outputs)
        return self.out_tensors


class ModelLayer(Layer):
    """
    The class :class:`ModelLayer` converts a :class:`Model` to a :class:`Layer` instance.

    Note that only a :class:`Model` with specified inputs and outputs can be converted to a :class:`ModelLayer`.
    For example, a customized model in dynamic eager mode normally does NOT have specified inputs and outputs so the
    customized model in dynamic eager mode can NOT be converted to a :class:`ModelLayer`.

    Parameters
    ----------
    model: tl.models.Model
        A model.
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.

    Methods
    ---------
    __init__()
        Initializing the ModelLayer.
    weights()
        Same as the weights of the given model.
    build()
        Do nothing because the given model has already been built.
    forward()
        Forward the computation. Simply call the forward() of the given model.
    """

    def __init__(self, model, name=None):
        """
        Initializing the ModelLayer given a instance of Model.

        :param model:  tl.models.Model
        """
        super(ModelLayer, self).__init__(name=name)

        self.model = model

        # Layer building state
        self._built = True

        # Layer weight state
        self._all_weights = model.all_weights

        # Layer training state
        self.is_train = True

        logging.info("ModelLayer %s from Model: %s" % (self.name, self.model.name))

    def __repr__(self):
        tmpstr = 'ModelLayer' + '(\n'

        modstr = self.model.__repr__()
        modstr = _addindent(modstr, 2)

        tmpstr += modstr + ')'
        return tmpstr

    def build(self, inputs_shape):
        pass

    def forward(self, inputs):
        return self.model.forward(inputs)

    def _set_mode_for_layers(self, is_train):
        """ Set training/evaluation mode for the ModelLayer."""
        self.is_train = is_train
        return self.model._set_mode_for_layers(is_train)

    def _fix_nodes_for_layers(self):
        """ fix LayerNodes to stop growing for this ModelLayer."""
        self._nodes_fixed = True
        self.model._fix_nodes_for_layers()

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted in order to release memory.
        """

        super(ModelLayer, self)._release_memory()
        self.model.release_memory()

    def get_args(self):
        init_args = {}
        init_args.update({"layer_type": "modellayer"})
        init_args["model"] = utils.net2static_graph(self.layer_args["model"])
        return init_args


class LayerList(Layer):
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
        self.layers = layers

        is_built = True
        for layer in self.layers:
            if layer._built is False:
                is_built = False
            if layer._built and layer.all_weights is not None:
                # some layers in the list passed in have already been built
                # e.g. using input shape to construct layers in dynamic eager
                if self._all_weights is None:
                    self._all_weights = list()
                self._all_weights.extend(layer.all_weights)
        if is_built:
            self._built = True

        logging.info(
            "LayerList %s including layers [%s]" % (self.name, ', '.join([layer.name for layer in self.layers]))
        )

        # check layer name uniqueness in LayerList
        local_layer_name_set = set()
        for layer in self.layers:
            if layer.name not in local_layer_name_set:
                local_layer_name_set.add(layer.name)
            else:
                raise ValueError(
                    'Layer name \'%s\' has already been used by another layer. Please change the layer name.' %
                    layer.name
                )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LayerList(list(self.layers)[idx])
        else:
            return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        tmpstr = 'LayerList' + '(\n'
        for idx, layer in enumerate(self.layers):
            modstr = layer.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + str(idx) + '): ' + modstr + '\n'

        tmpstr = tmpstr + ')'
        return tmpstr

    def build(self, inputs_shape):
        """
        Build the LayerList. The layer instances will be connected automatically one by one.
        """
        in_tensor = self._input_tensors
        # in_layer = self._input_layer
        for layer in self.layers:
            is_build = layer._built
            out_tensor = layer(in_tensor)
            # nlayer = layer(in_layer)
            if is_build is False and layer.all_weights is not None:
                if self._all_weights is None:
                    self._all_weights = list()
                self._all_weights.extend(layer.all_weights)
            layer._built = True
            in_tensor = out_tensor
            # in_layer = nlayer

    def forward(self, inputs):
        """
        Forward the computation. The computation will go through all layer instances.
        """
        z = inputs
        for layer in self.layers:
            z = layer.forward(z)
        return z

    def _set_mode_for_layers(self, is_train):
        """Set training/evaluation mode for all layer instances."""
        self.is_train = is_train
        for layer in self.layers:
            if isinstance(layer, ModelLayer):
                layer._set_mode_for_layers(is_train)
            elif isinstance(layer, LayerList):
                layer._set_mode_for_layers(is_train)
            else:
                layer.is_train = is_train

    def _fix_nodes_for_layers(self):
        """ fix LayerNodes to stop growing for this LayerList."""
        self._nodes_fixed = True
        for layer in self.layers:
            layer._fix_nodes_for_layers()

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted.
        """
        super(LayerList, self)._release_memory()
        for layer in self.layers:
            layer._release_memory()

    def get_args(self):
        init_args = {}
        layers = self.layer_args["layers"]
        init_args["layers"] = [layer.config for layer in layers]
        init_args.update({"layer_type": "layerlist"})
        return init_args


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def tolist(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        ntensors = list()
        for t in tensors:
            ntensors += tolist(t)
        return ntensors
    else:
        return [tensors]
