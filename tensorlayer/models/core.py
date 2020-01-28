import os
from abc import abstractmethod
from queue import Queue

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files import utils
from tensorlayer.layers import Layer, ModelLayer

__all__ = [
    'Model',
]

_global_model_name_dict = {}  # TODO: better implementation?
_global_model_name_set = set()


class Model(object):
    """The :class:`Model` class represents a neural network.

    It should be subclassed when implementing a dynamic model,
    where 'forward' method must be overwritten.
    Otherwise, please specify 'inputs' tensor(s) and 'outputs' tensor(s)
    to create a static model. In that case, 'inputs' tensors should come
    from tl.layers.Input().

    Parameters
    -----------
    inputs : a Layer or list of Layer
        The input(s) to the model.
    outputs : a Layer or list of Layer
        The output(s) to the model.
    name : None or str
        The name of the model.

    Methods
    ---------
    __init__(self, inputs=None, outputs=None, name=None)
        Initializing the Model.
    inputs()
        Get input tensors to this network (only avaiable for static model).
    outputs()
        Get output tensors to this network (only avaiable for static model).
    __call__(inputs, is_train=None, **kwargs)
        Forward input tensors through this network.
    all_layers()
        Get all layer objects of this network in a list of layers.
    weights()
        Get the weights of this network in a list of tensors.
    train()
        Set this network in training mode. (affect layers e.g. Dropout, BatchNorm).
    eval()
        Set this network in evaluation mode.
    as_layer()
        Set this network as a ModelLayer so that it can be integrated into another Model.
    release_memory()
        Release the memory that was taken up by tensors which are maintained by this network.
    save_weights(self, filepath, format='hdf5')
        Save the weights of this network in a given format.
    load_weights(self, filepath, format=None, in_order=True, skip=False)
        Load weights into this network from a specified file.
    save(self, filepath, save_weights=True)
        Save the network with/without weights.
    load(filepath, save_weights=True)
        Load the network with/without weights.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from tensorlayer.layers import Input, Dense, Dropout
    >>> from tensorlayer.models import Model

    Define static model

    >>> class CustomModel(Model):
    >>>     def __init__(self):
    >>>         super(CustomModel, self).__init__()
    >>>         self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)
    >>>         self.dropout1 = Dropout(keep=0.8)
    >>>         self.dense2 = Dense(n_units=10, in_channels=800)
    >>>     def forward(self, x):
    >>>         z = self.dense1(x)
    >>>         z = self.dropout1(z)
    >>>         z = self.dense2(z)
    >>>         return z
    >>> M_dynamic = CustomModel()

    Define static model

    >>> ni = Input([None, 784])
    >>> nn = Dense(n_units=800, act=tf.nn.relu)(ni)
    >>> nn = Dropout(keep=0.8)(nn)
    >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
    >>> M_static = Model(inputs=ni, outputs=nn, name="mlp")

    Get network information

    >>> print(M_static)
    ... Model(
    ...  (_inputlayer): Input(shape=[None, 784], name='_inputlayer')
    ...  (dense): Dense(n_units=800, relu, in_channels='784', name='dense')
    ...  (dropout): Dropout(keep=0.8, name='dropout')
    ...  (dense_1): Dense(n_units=10, relu, in_channels='800', name='dense_1')
    ... )

    Forwarding through this network

    >>> data = np.random.normal(size=[16, 784]).astype(np.float32)
    >>> outputs_d = M_dynamic(data)
    >>> outputs_s = M_static(data)

    Save and load weights

    >>> M_static.save_weights('./model_weights.h5')
    >>> M_static.load_weights('./model_weights.h5')

    Save and load the model

    >>> M_static.save('./model.h5')
    >>> M = Model.load('./model.h5')

    Convert model to layer

    >>> M_layer = M_static.as_layer()

    """

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs=None, outputs=None, name=None):
        """
        Initializing the Model.

        Parameters
        ----------
        inputs : Tensor or list of tensors
            Input tensor(s), which must come from tl.layers.Input()
        outputs : Tensor or list of tensors
            Output tensor(s), which must be the output(s) of some TL layers
        name : str or None
            Name for this network
        """
        # Auto naming if the name is not given
        self._NameNone = False
        global _global_model_name_dict
        global _global_model_name_set
        if name is None:
            self._NameNone = True
            prefix = self.__class__.__name__.lower()
            if _global_model_name_dict.get(prefix) is not None:
                _global_model_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_model_name_dict[prefix])
            else:
                _global_model_name_dict[prefix] = 0
                name = prefix
            while name in _global_model_name_set:
                _global_model_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_model_name_dict[prefix])
            _global_model_name_set.add(name)
        else:
            if name in _global_model_name_set:
                raise ValueError(
                    'Model name \'%s\' has already been used by another model. Please change the model name.' % name
                )
            _global_model_name_set.add(name)
            _global_model_name_dict[name] = 0

        # Model properties
        self.name = name

        # Model state: train or test
        self.is_train = None

        # Model weights
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # Model args of all layers, ordered by all_layers
        self._config = None

        # Model inputs and outputs
        # TODO: note that in dynamic network, inputs and outputs are both None, may cause problem, test needed
        self._inputs = inputs
        self._outputs = outputs

        # Model converted into a Layer
        self._model_layer = None

        # Layer Node status
        self._nodes_fixed = False

        # Model layers
        self._all_layers = None

        if inputs is None and outputs is None:
            pass

        else:
            # check type of inputs and outputs
            check_order = ['inputs', 'outputs']
            for co, check_argu in enumerate([inputs, outputs]):
                if isinstance(check_argu, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(check_argu):
                    pass
                elif isinstance(check_argu, list):
                    if len(check_argu) == 0:
                        raise ValueError(
                            "The argument `%s` is detected as an empty list. " % check_order[co] +
                            "It should be either Tensor or a list of Tensor."
                        )
                    for idx in range(len(check_argu)):
                        if not isinstance(check_argu[idx], tf_ops._TensorLike) or not tf_ops.is_dense_tensor_like(
                                check_argu[idx]):
                            raise TypeError(
                                "The argument `%s` should be either Tensor or a list of Tensor " % (check_order[co]) +
                                "but the %s[%d] is detected as %s" % (check_order[co], idx, type(check_argu[idx]))
                            )
                else:
                    raise TypeError(
                        "The argument `%s` should be either Tensor or a list of Tensor but received %s" %
                        (check_order[co], type(check_argu))
                    )

            if not _check_tl_layer_tensors(inputs):
                raise TypeError(
                    "The argument `inputs` should be either Tensor or a list of Tensor "
                    "that come from TensorLayer's Input layer: tl.layers.Input(shape). "
                )
            if not _check_tl_layer_tensors(outputs):
                raise TypeError(
                    "The argument `outputs` should be either Tensor or a list of Tensor "
                    "that is/are outputs from some TensorLayer's layers, e.g. tl.layers.Dense, tl.layers.Conv2d."
                )

            # build network graph
            self._node_by_depth, self._all_layers = self._construct_graph()

            self._fix_nodes_for_layers()

    def __call__(self, inputs, is_train=None, **kwargs):
        """Forward input tensors through this network by calling.

        Parameters
        ----------
        inputs : Tensor or list of Tensors, numpy.ndarray of list of numpy.ndarray
            Inputs for network forwarding
        is_train : boolean
            Network's mode for this time forwarding. If 'is_train' == True, this network is set as training mode.
            If 'is_train' == False, this network is set as evaluation mode
        kwargs :
            For other keyword-only arguments.

        """

        self._check_mode(is_train)

        # FIXME: this may cause inefficiency, this is used to check if every layer is built
        self.all_layers

        # fix LayerNodes when first calling
        if self._nodes_fixed is False:
            self._fix_nodes_for_layers()

        # set training / inference mode if necessary
        if is_train is not None:
            self._set_mode_for_layers(is_train)

        # if self._input is a list, then it must be a static network
        if isinstance(self._inputs, list):
            if not isinstance(inputs, list):
                raise ValueError("The argument `inputs` should be a list of values but detected as %s." % type(inputs))
            elif len(inputs) != len(self._inputs):
                raise ValueError(
                    "The argument `inputs` should be a list with len=%d but detected as len=%d." %
                    (len(self._inputs), len(inputs))
                )

        # convert inputs to tensor if it is originally not
        # FIXME: not sure convert_to_tensor here or ask user to do it
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                inputs[idx] = tf.convert_to_tensor(inputs[idx])
        else:
            inputs = tf.convert_to_tensor(inputs)

        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Network forwarding given input tensors

        Parameters
        ----------
        inputs : Tensor or list of Tensors
            input tensor(s)
        kwargs :
            For other keyword-only arguments.

        Returns
        -------
            output tensor(s) : Tensor or list of Tensor(s)

        """
        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is None:
            raise ValueError(
                "Outputs not defined. Please define inputs and outputs when the model is created. Or overwrite forward() function."
            )

        memory = dict()

        # get each layer's output by going through the graph in depth order
        for depth, nodes in enumerate(self._node_by_depth):
            if depth == 0:
                if isinstance(self.inputs, list):
                    assert len(inputs[0]) == len(nodes)
                    for idx, node in enumerate(nodes):
                        memory[node.name] = node(inputs[0][idx])
                else:
                    memory[nodes[0].name] = nodes[0](inputs[0])
            else:
                for node in nodes:
                    in_nodes = node.in_nodes
                    in_tensors_idxes = node.in_tensors_idxes
                    if len(in_nodes) == 1:
                        node_input = memory[in_nodes[0].name][in_tensors_idxes[0]]
                    else:
                        node_input = [memory[inode.name][idx] for inode, idx in zip(in_nodes, in_tensors_idxes)]
                    memory[node.name] = node(node_input)

        if not isinstance(self._outputs, list):
            return memory[self._outputs._info[0].name][self._outputs._info[1]]
        else:
            return [memory[tensor._info[0].name][tensor._info[1]] for tensor in self._outputs]

    @property
    def all_layers(self):
        """Return all layers of this network in a list."""
        if self._all_layers is not None:
            return self._all_layers

        if self._inputs is not None and self._outputs is not None:
            # static model
            return self._all_layers
        else:
            # dynamic model
            self._all_layers = list()
            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("all_weights")
            attr_list.remove("trainable_weights")
            attr_list.remove("nontrainable_weights")
            attr_list.remove("_all_weights")
            attr_list.remove("_trainable_weights")
            attr_list.remove("_nontrainable_weights")
            attr_list.remove("all_layers")
            attr_list.remove("_all_layers")
            attr_list.remove("n_weights")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), Layer):
                        nowlayer = getattr(self, attr)
                        if not nowlayer._built:
                            raise AttributeError("Layer %s not built yet." % repr(nowlayer))
                        self._all_layers.append(nowlayer)
                    elif isinstance(getattr(self, attr), Model):
                        nowmodel = getattr(self, attr)
                        self._all_layers.append(nowmodel)
                    elif isinstance(getattr(self, attr), list):
                        self._all_layers.extend(_add_list_to_all_layers(getattr(self, attr)))
                # TODO: define customised exception for TL
                except AttributeError as e:
                    raise e
                except Exception:
                    pass

            # check layer name uniqueness
            local_layer_name_dict = set()
            for layer in self._all_layers:
                if layer.name in local_layer_name_dict:
                    raise ValueError(
                        'Layer name \'%s\' has already been used by another layer. Please change the layer name.' %
                        layer.name
                    )
                else:
                    local_layer_name_dict.add(layer.name)
            return self._all_layers

    @property
    def trainable_weights(self):
        """Return trainable weights of this network in a list."""
        if self._trainable_weights is not None and len(self._trainable_weights) > 0:
            # self._trainable_weights already extracted, so do nothing
            pass
        else:
            self._trainable_weights = []
            for layer in self.all_layers:
                if layer.trainable_weights is not None:
                    self._trainable_weights.extend(layer.trainable_weights)

        return self._trainable_weights.copy()

    @property
    def nontrainable_weights(self):
        """Return nontrainable weights of this network in a list."""
        if self._nontrainable_weights is not None and len(self._nontrainable_weights) > 0:
            # self._nontrainable_weights already extracted, so do nothing
            pass
        else:
            self._nontrainable_weights = []
            for layer in self.all_layers:
                if layer.nontrainable_weights is not None:
                    self._nontrainable_weights.extend(layer.nontrainable_weights)

        return self._nontrainable_weights.copy()

    @property
    def all_weights(self):
        """Return all weights of this network in a list."""
        if self._all_weights is not None and len(self._all_weights) > 0:
            # self._all_weights already extracted, so do nothing
            pass
        else:
            self._all_weights = []
            for layer in self.all_layers:
                if layer.all_weights is not None:
                    self._all_weights.extend(layer.all_weights)

        return self._all_weights.copy()

    @property
    def n_weights(self):
        """Return the number of weights (parameters) in this network."""
        n_weights = 0
        for i, w in enumerate(self.all_weights):
            n = 1
            # for s in p.eval().shape:
            for s in w.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_weights = n_weights + n
        # print("num of weights (parameters) %d" % n_weights)
        return n_weights

    @property
    def config(self):
        if self._config is not None and len(self._config) > 0:
            return self._config
        else:
            # _config = []
            _config = {}
            if self._NameNone is True:
                _config.update({"name": None})
            else:
                _config.update({"name": self.name})
            version_info = {
                "tensorlayer_version": tl.__version__,
                "backend": "tensorflow",
                "backend_version": tf.__version__,
                "training_device": "gpu",
                "save_date": None,
            }
            _config["version_info"] = version_info
            # if self.outputs is None:
            #     raise RuntimeError(
            #         "Dynamic mode does not support config yet."
            #     )
            model_architecture = []
            for layer in self.all_layers:
                model_architecture.append(layer.config)
            _config["model_architecture"] = model_architecture
            if self.inputs is not None:
                if not isinstance(self.inputs, list):
                    _config.update({"inputs": self.inputs._info[0].name})
                else:
                    config_inputs = []
                    for config_input in self.inputs:
                        config_inputs.append(config_input._info[0].name)
                    _config.update({"inputs": config_inputs})
            if self.outputs is not None:
                if not isinstance(self.outputs, list):
                    _config.update({"outputs": self.outputs._info[0].name})
                else:
                    config_outputs = []
                    for config_output in self.outputs:
                        config_outputs.append(config_output._info[0].name)
                    _config.update({"outputs": config_outputs})
            if self._nodes_fixed or self.outputs is None:
                self._config = _config

            return _config

    def train(self):
        """Set this network in training mode. After calling this method,
        all layers in network are in training mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.models.vgg16()
        >>> net.train()

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
        >>> net = tl.models.vgg16()
        >>> net.eval()
        # do evaluation

        """
        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    def test(self):
        """Set this network in evaluation mode."""
        self.eval()

    def infer(self):
        """Set this network in evaluation mode."""
        self.eval()

    def as_layer(self):
        """Return this network as a ModelLayer so that it can be integrated into another Model.

        Examples
        --------
        >>> from tensorlayer.layers import Input, Dense, Dropout
        >>> from tensorlayer.models import Model
        >>> ni = Input([None, 784])
        >>> nn = Dense(n_units=800, act=tf.nn.relu)(ni)
        >>> nn = Dropout(keep=0.8)(nn)
        >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
        >>> M_hidden = Model(inputs=ni, outputs=nn, name="mlp").as_layer()
        >>> nn = M_hidden(ni)   # use previously constructed model as layer
        >>> nn = Dropout(keep=0.8)(nn)
        >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
        >>> M_full = Model(inputs=ni, outputs=nn, name="mlp")

        """
        if self._outputs is None:
            raise AttributeError("Dynamic network cannot be converted to Layer.")

        if self._model_layer is None:
            self._model_layer = ModelLayer(self)

        return self._model_layer

    def _check_mode(self, is_train):
        """Check whether this network is in a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """
        # contradiction test
        if is_train is None and self.is_train is None:
            raise ValueError(
                "Training / inference mode not defined. Argument `is_train` should be set as True / False. Otherwise please use `Model.train()` / `Model.eval()` to switch the mode."
            )
        elif is_train is not None and self.is_train is not None:
            if is_train == self.is_train:
                logging.warning(
                    "Training / inference mode redefined redundantly. Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode."
                )
            else:
                raise AttributeError(
                    "Training / inference mode mismatch. The argument `is_train` is set as %s, " % is_train +
                    "but the mode is currently set as %s. " %
                    ('Training by Model.train()' if self.is_train else 'Inference by Model.eval()') +
                    "Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode."
                )

    def _set_mode_for_layers(self, is_train):
        """Set all layers of this network to a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """
        for layer in self.all_layers:
            if isinstance(layer, Model):
                layer.is_train = is_train
            layer._set_mode_for_layers(is_train)

    def _fix_nodes_for_layers(self):
        """Fix each Layer's LayerNode to stop growing, see LayerNode for more."""
        for layer in self.all_layers:
            layer._fix_nodes_for_layers()
        self._nodes_fixed = True

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            if value._built is False:
                raise AttributeError(
                    "The registered layer `{}` should be built in advance. "
                    "Do you forget to pass the keyword argument 'in_channels'? ".format(value.name)
                )
        super().__setattr__(key, value)

    def __repr__(self):
        # tmpstr = self.__class__.__name__ + '(\n'
        tmpstr = self.name + '(\n'
        for idx, layer in enumerate(self.all_layers):
            modstr = layer.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + layer.name + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    ## raise Exceptions for old version codes
    def print_all_layers(self):
        raise Exception("please change net.print_all_layers --> print(net)")

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

    def get_layer(self, name=None, index=None):
        """Network forwarding given input tensors

        Parameters
        ----------
        name : str or None
            Name of the requested layer. Default None.
        index : int or None
            Index of the requested layer. Default None.

        Returns
        -------
            layer : The requested layer

        Notes
        -----
        Either a layer name or a layer index should be given.

        """
        if index is not None:
            if len(self.all_layers) <= index:
                raise ValueError(
                    'model only has ' + str(len(self.all_layers)) + ' layers, but ' + str(index) +
                    '-th layer is requested.'
                )
            else:
                return self.all_layers[index]
        elif name is not None:
            for layer in self.all_layers:
                if layer.name == name:
                    return layer
            raise ValueError('Model has no layer named ' + name + '.')
        else:
            raise ValueError('Either a layer name or a layer index should be given.')

    def _construct_graph(self):
        """construct computation graph for static model using LayerNode object"""
        all_layers = []
        node_by_depth = []  # [[node0, node1], [node2, node3], ...]

        input_tensors_list = self.inputs if isinstance(self.inputs, list) else [self.inputs]

        queue_node = Queue()

        # BFS to visit all nodes that should be involved in the computation graph
        output_tensors_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        output_nodes = [tensor._info[0] for tensor in output_tensors_list]

        visited_node_names = set()
        for out_node in output_nodes:
            if out_node.visited:
                continue
            queue_node.put(out_node)

            while not queue_node.empty():
                cur_node = queue_node.get()
                in_nodes = cur_node.in_nodes

                for node in in_nodes:
                    node.out_nodes.append(cur_node)
                    if not node.visited:
                        queue_node.put(node)
                        node.visited = True
                        if node.name not in visited_node_names:
                            visited_node_names.add(node.name)
                        # else have multiple layers with the same name
                        else:
                            raise ValueError(
                                'Layer name \'%s\' has already been used by another layer. Please change the layer name.'
                                % node.layer.name
                            )

        # construct the computation graph in top-sort order
        cur_depth = [tensor._info[0] for tensor in input_tensors_list]
        next_depth = []
        indegrees = {}

        visited_layer_names = []
        while not len(cur_depth) == 0:
            node_by_depth.append(cur_depth)
            for node in cur_depth:
                if node.layer.name not in visited_layer_names:
                    all_layers.append(node.layer)
                    visited_layer_names.append(node.layer.name)
                for out_node in node.out_nodes:
                    if out_node.name not in indegrees.keys():
                        indegrees[out_node.name] = len(out_node.in_nodes)
                    indegrees[out_node.name] -= 1
                    if indegrees[out_node.name] == 0:
                        next_depth.append(out_node)

            cur_depth = next_depth
            next_depth = []

        return node_by_depth, all_layers

    def release_memory(self):
        '''
        WARNING: This function should be called with great caution.

        Release objects that MAY NOT be necessary such as layer.outputs (if in a tf.GradientTape() scope).
        For each layer in the model, layer.inputs and layer.outputs will be set as None but not deleted.

        A void function.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> vgg = tl.models.vgg16()
        ... # training preparation
        ... # ...
        ... # back propagation
        >>> with tf.GradientTape() as tape:
        >>>     _logits = vgg(x_batch)
        >>>     ## compute loss and update model
        >>>     _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
        >>>     ## release unnecessary objects (layer.inputs, layer.outputs)
        >>>     ## this function should be called with great caution
        >>>     ## within the scope of tf.GradientTape(), using this function should be fine
        >>>     vgg.release_memory()

        '''
        for layer in self.all_layers:
            layer._release_memory()

    def save(self, filepath, save_weights=True, customized_data=None):
        """
        Save model into a given file.
        This function save can save both the architecture of neural networks and weights (optional).
        WARNING: If the model contains Lambda / ElementwiseLambda layer, please check the documentation of Lambda / ElementwiseLambda layer and find out the cases that have / have not been supported by Model.save().

        Parameters
        ----------
        filepath : str
            Filename into which the model will be saved.
        save_weights : bool
            Whether to save model weights.
        customized_data : dict
            The user customized meta data.

        Examples
        --------
        >>> net = tl.models.vgg16()
        >>> net.save('./model.h5', save_weights=True)
        >>> new_net = Model.load('./model.h5', load_weights=True)

        """
        # TODO: support saving LambdaLayer that includes parametric self defined function with outside variables
        if self.outputs is None:
            raise RuntimeError(
                "Model save() not support dynamic mode yet.\nHint: you can use Model save_weights() to save the weights in dynamic mode."
            )
        utils.save_hdf5_graph(
            network=self, filepath=filepath, save_weights=save_weights, customized_data=customized_data
        )

    @staticmethod
    def load(filepath, load_weights=True):
        """
        Load model from a given file, which should be previously saved by Model.save().
        This function load can load both the architecture of neural networks and weights (optional, and needs to be saved in Model.save()).
        When a model is loaded by this function load, there is no need to reimplement or declare the architecture of the model explicitly in code.
        WARNING: If the model contains Lambda / ElementwiseLambda layer, please check the documentation of Lambda / ElementwiseLambda layer and find out the cases that have / have not been supported by Model.load().

        Parameters
        ----------
        filepath : str
            Filename from which the model will be loaded.
        load_weights : bool
            Whether to load model weights.

        Examples
        --------
        >>> net = tl.models.vgg16()
        >>> net.save('./model.h5', save_weights=True)
        >>> new_net = Model.load('./model.h5', load_weights=True)
        """
        # TODO: support loading LambdaLayer that includes parametric self defined function with outside variables
        M = utils.load_hdf5_graph(filepath=filepath, load_weights=load_weights)
        return M

    def save_weights(self, filepath, format=None):
        """Input filepath, save model weights into a file of given format.
            Use self.load_weights() to restore.

        Parameters
        ----------
        filepath : str
            Filename to which the model weights will be saved.
        format : str or None
            Saved file format.
            Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            1) If this is set to None, then the postfix of filepath will be used to decide saved format.
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
        >>> net = tl.models.vgg16()
        >>> net.save_weights('./model.h5')
        ...
        >>> net.load_weights('./model.h5')

        2) Save model weights in npz/npz_dict format
        >>> net = tl.models.vgg16()
        >>> net.save_weights('./model.npz')
        >>> net.save_weights('./model.npz', format='npz_dict')

        """
        if self.all_weights is None or len(self.all_weights) == 0:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format is None:
            postfix = filepath.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
                format = postfix
            else:
                format = 'hdf5'

        if format == 'hdf5' or format == 'h5':
            utils.save_weights_to_hdf5(filepath, self)
        elif format == 'npz':
            utils.save_npz(self.all_weights, filepath)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.all_weights, filepath)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                "Other format is not supported now."
            )

    def load_weights(self, filepath, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        filepath : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the filepath will be used to decide its format. If specified,
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
        >>> net = tl.models.vgg16()
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
        if not os.path.exists(filepath):
            raise FileNotFoundError("file {} doesn't exist.".format(filepath))

        if format is None:
            format = filepath.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            if skip ==True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(filepath, self, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(filepath, self)
        elif format == 'npz':
            utils.load_and_assign_npz(filepath, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(filepath, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                "Other format is not supported now."
            )

    # TODO: not supported now
    # def save_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', global_step=None, printable=False):
    #     # TODO: Documentation pending
    #     """"""
    #     if not os.path.exists(save_dir):
    #         raise FileNotFoundError("Save directory {} doesn't exist.".format(save_dir))
    #     utils.save_ckpt(sess, mode_name, save_dir, self.weights, global_step, printable)
    #
    # def load_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', is_latest=True, printable=False):
    #     # TODO: Documentation pending
    #     """"""
    #     utils.load_ckpt(sess, mode_name, save_dir, self.weights, is_latest, printable)


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


def _check_tl_layer_tensors(tensors):
    if not isinstance(tensors, list):
        return hasattr(tensors, '_info')
    else:
        for t in tensors:
            if not hasattr(t, '_info'):
                return False
        return True


def _add_list_to_all_layers(list_member):
    temp_all_layers = list()
    for component in list_member:
        if isinstance(component, Layer):
            temp_all_layers.append(component)
            if not component._built:
                raise AttributeError("Layer %s not built yet." % repr(component))
        elif isinstance(component, Model):
            temp_all_layers.append(component)
        elif isinstance(component, list):
            temp_all_layers.extend(_add_list_to_all_layers(component))
    return temp_all_layers
