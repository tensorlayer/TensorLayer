#! /usr/bin/python
# -*- coding: utf-8 -*-
from .common import str2act
from tensorlayer.backend.ops.load_backend import BACKEND
from collections import OrderedDict
import time, os
import tensorlayer as tl
from tensorlayer.decorators import (protected_method)
from tensorlayer.files import utils
from tensorlayer.layers.utils import (get_variable_with_initializer)
from tensorlayer import logging

_global_layer_name_dict = {}  # TODO: better implementation?

if BACKEND == 'tensorflow':
    import tensorflow as tf
    Parameter_ = tf.Variable
elif BACKEND == 'dragon':
    import dragon as dg
    Parameter_ = dg.Tensor  # TODO the dragon parameter is a initializers
else:
    raise NotImplementedError("This backend is not supported")


class Module(object):

    def __init__(self, name=None, act=None, *args, **kwargs):
        self._params = OrderedDict()
        self._layers = OrderedDict()
        self._params_status = OrderedDict()
        self._parameter_layout_dict = {}
        self._create_time = int(time.time() * 1e9)

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
            self.act = str2act(act)
        else:
            if act:
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

        # nested layers
        # self._layers = None

        # Layer training state
        self.is_train = True

    def extend_repr(self):
        """
        Sets the extended representation of the Cell.

        To print customized extended information, re-implement this method in your own cells.
        """
        return ''

    def __repr__(self):
        extra_str = self.extend_repr()
        info_str = self.__class__.__name__ + '<'
        if self._layers:
            sub_str = '\n'
            if extra_str:
                sub_str += '{}\n'.format(self.extend_repr())
            for key, value in self._layers.items():
                sub_str += '({}): {}\n'.format(key, repr(value))
            sub_str = sub_str.replace('\n', '\n  ') + '>'
            info_str += sub_str
        else:
            info_str += extra_str + '>'
        return info_str

    def __setattr__(self, name, value):
        layers = self.__dict__.get('_layers')
        params = self.__dict__.get('_params')

        if isinstance(value, Parameter_):
            if params is None:
                raise AttributeError("Can not assign params before Module.__init__() call.")
            if name in self.__dict__:
                if self.__dict__[name] is not None:
                    raise TypeError("Expected type is not in (Parameter, Module), but got Parameter.")
                del self.__dict__[name]
            if layers and name in layers:
                raise TypeError("Expected type is Module, but got Parameter.")
            self.insert_param_to_layer(name, value)

        elif isinstance(value, Module):
            if layers is None:
                raise AttributeError("Can not assign layers before Module.__init__() call.")
            if name in self.__dict__:
                del self.__dict__[name]
            if params and name in params:
                raise TypeError("Expected type is Parameter, but got Module.")
            if value._built is False:
                raise AttributeError(
                    "The registered layer `{}` should be built in advance. "
                    "Do you forget to pass the keyword argument 'in_channels'? ".format(value.name)
                )
            layers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __call__(self, *inputs, **kwargs):
        if BACKEND in ['tensorflow', 'dragon']:
            output = self.forward(*inputs)
        else:
            exit("Unsupported backend")
        return output

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

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
        """Input file_path, save model weights into a file of given format.
                    Use self.load_weights() to restore.

                Parameters
                ----------
                file_path : str
                    Filename to which the model weights will be saved.
                format : str or None
                    Saved file format.
                    Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
                    1) If this is set to None, then the postfix of file_path will be used to decide saved format.
                    If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
                    2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
                    the hdf5 file.
                    3) 'npz' will save model weights sequentially into a npz file.
                    4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
                    5) 'ckpt' will save model weights into a tensorflow ckpt file.

                    Default None.

                Examples
                --------
                1) Save model weights in hdf5 format by default.
                >>> net = vgg16()
                >>> net.save_weights('./model.h5')
                ...
                >>> net.load_weights('./model.h5')

                2) Save model weights in npz/npz_dict format
                >>> net = vgg16()
                >>> net.save_weights('./model.npz')
                >>> net.save_weights('./model.npz', format='npz_dict')

                """

        # self.all_weights = self.network.all_weights
        if self.all_weights is None or len(self.all_weights) == 0:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format is None:
            postfix = file_path.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
                format = postfix
            else:
                format = 'hdf5'

        if format == 'hdf5' or format == 'h5':
            utils.save_weights_to_hdf5(file_path, self)
        elif format == 'npz':
            utils.save_npz(self.all_weights, file_path)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.all_weights, file_path)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                "Other format is not supported now."
            )

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        file_path : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
            value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is None.
        in_order : bool
            Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
            If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
            If 'in_order' is False, weights from the file will be loaded into model by matching the name
            with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
            a weights file which is saved in graph(eager) mode.
            Default is True.
        skip : bool
            Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
            'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
            whose name is not found in model weights (self.all_weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file.
        >>> net = vgg16()
        >>> net.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> net.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file
        >>> net.load_weights('./model.npz')

        2) load model from a npz file, which is saved as npz_dict previously
        >>> net.load_weights('./model.npz', format='npz_dict')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("file {} doesn't exist.".format(file_path))

        if format is None:
            format = file_path.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            if skip ==True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(file_path, self, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(file_path, self)
        elif format == 'npz':
            utils.load_and_assign_npz(file_path, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(file_path, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                "Other format is not supported now."
            )

    def _set_mode_for_layers(self, is_train):
        """Set all layers of this network to a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """
        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            if isinstance(layer, Module):
                layer.is_train = is_train

    def set_train(self):
        """Set this network in training mode. After calling this method,
        all layers in network are in training mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.vgg16()
        >>> net.set_train()

        """
        if self.is_train !=True:
            self.is_train = True
            self._set_mode_for_layers(True)

    def eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.vgg16()
        >>> net.eval()
        # do evaluation

        """
        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tl.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tl.get_tensor_shape(tensors)
        return shape_mem

    def insert_param_to_layer(self, param_name, param, check_name=True):
        """
        Adds a parameter to the current layer.

        Inserts a parameter with given name to the layer. Please refer to the usage in
        source code of `tensorlayer.layer.Module.__setattr__`.

        Args:
            param_name (str): Name of the parameter.
            param (Parameter): Parameter to be inserted to the layer.
            check_name (bool): Determines whether the name input is compatible. Default: True.

        Raises:
            KeyError: If the name of parameter is null or contains dot.
            AttributeError: If user did not call init() first.
            TypeError: If the type of parameter is not Parameter.
        """
        if not param_name:
            raise KeyError("The name of parameter should not be null.")
        if check_name and '.' in param_name:
            raise KeyError("The name of parameter should not contain \".\"")
        if '_params' not in self.__dict__:
            raise AttributeError("You need call init() first.")
        if hasattr(self, param_name) and param_name not in self._params:
            raise KeyError("Duplicated parameter name '{}'.".format(param_name))
        if not isinstance(param, Parameter_) and param is not None:
            raise TypeError("The type of parameter should be 'Parameter' if not None.")
        self._params[param_name] = param
        try:
            self._params_status[param_name] = self.trainable
        except:
            pass

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

    @property
    def create_time(self):
        return self._create_time

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                return params[name]
        if '_layers' in self.__dict__:
            layers = self.__dict__['_layers']
            if name in layers:
                return layers[name]
        if '_params_status' in self.__dict__:
            params_status = self.__dict__['_params_status']
            if name in params_status:
                return params_status[name]
        raise AttributeError("'{}' object has no attribute '{}'.".format(type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._layers:
            del self._layers[name]
        else:
            object.__delattr__(self, name)

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
        self.get_weights()
        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            params = layer._params.items()
            params_status = layer._params_status.items()
            params_zip = zip(params, params_status)
            for params, params_status in params_zip:
                if params_status[1] ==True:
                    self._trainable_weights.append(params[1])
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
        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            params = layer._params.items()
            params_status = layer._params_status.items()
            params_zip = zip(params, params_status)
            for params, params_status in params_zip:
                if params_status[1] == False:
                    self._nontrainable_weights.append(params[1])
        return self._nontrainable_weights

    @property
    def all_weights(self):
        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            params = layer._params.items()
            for par, val in params:
                self._all_weights.append(val)
        return self._all_weights

    def get_weights(self, expand=True):
        """
        Returns an iterator over layer weights.

        Yields weights of this layer. If `expand` is True, yield parameters of this layer and all sublayers.

        Args:
            expand (bool): If True, yields parameters of this layer and all sublayers. Otherwise, yields only parameters
                           that are direct members of this layer. Default: True.

        Examples:
            >>> net = Net()
            >>> for item in net.get_weights():
            >>>     print(item)
        """
        for _, param in self.parameters_and_names(expand=expand):
            yield param

    def check_names(self):
        names = set("")
        for value, param in self.parameters_and_names():
            if param.name in names:
                raise ValueError(
                    "The value of {} is {}, its name '{}' already exists.".format(value, param, param.name)
                )
            names.add(param.name)

    def parameters_and_names(self, name_prefix='', expand=True):
        """
        Returns an iterator over layer parameters.

        Includes the parameter's name  and itself.

        Args:
            name_prefix (str): Namespace. Default: ''.
            expand (bool): If True, yields parameters of this layer and all sublayers. Otherwise, yields only parameters
                           that are direct members of this layer. Default: True.

        Examples:
            >>> n = Net()
            >>> names = []
            >>> for m in n.parameters_and_names():
            >>>     if m[0]:
            >>>         names.append(m[0])
        """
        layers = []
        if expand:
            layers = self.layers_and_names(name_prefix=name_prefix)
        else:
            layers.append((name_prefix, self))

        params_set = set()
        for layer_name, layer in layers:
            params = layer._params.items()
            for par_name, par in params:
                if par.inited_param is not None:
                    par = par.inited_param
                if par is not None and id(par) not in params_set:
                    params_set.add(id(par))
                    par_new_name = par_name
                    if layer_name:
                        par_new_name = layer_name + '.' + par_new_name

                    yield par_new_name, par

    def layers_and_names(self, layers=None, name_prefix=''):
        """
        Returns an iterator over all layers in the network.

        Includes the layer's name and itself.

        Args:
            layers (str): layers to iterate over. Default: None.
            name_prefix (str): Namespace. Default: ''.

        Examples:
            >>> n = Net()
            >>> names = []
            >>> for m in n.layers_and_names():
            >>>     if m[0]:
            >>>         names.append(m[0])
        """
        t_layers = layers if layers else set()
        if self in t_layers:
            return

        t_layers.add(self)
        yield name_prefix, self

        for name, layer in self._layers.items():
            if layer:
                layers_name_prefix = name
                if name_prefix:
                    layers_name_prefix = name_prefix + '.' + layers_name_prefix
                for ele in layer.layers_and_names(t_layers, layers_name_prefix):
                    yield ele

    def layers(self):
        """Returns an iterator over immediate layers."""
        return self.name_layers().values()

    def name_layers(self):
        """
        Returns an iterator over all layers in the network.

        Include name of the layer and layer itself.
        """
        value_set = set()
        layers = OrderedDict()
        for name, layer in self._layers.items():
            if layer is not None and layer not in value_set:
                value_set.add(layer)
                layers[name] = layer
        return layers


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


def tolist(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        ntensors = list()
        for t in tensors:
            ntensors += tolist(t)
        return ntensors
    else:
        return [tensors]
