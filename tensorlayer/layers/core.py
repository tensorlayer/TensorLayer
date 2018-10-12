#! /usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import inspect
import six

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tensorlayer import logging

from tensorlayer.decorators import deprecated
from tensorlayer.decorators import layer_autoregister
from tensorlayer.decorators import overwrite_layername_in_network
from tensorlayer.decorators import protected_method
from tensorlayer.decorators import private_method

__all__ = ['LayersConfig', 'TF_GRAPHKEYS_VARIABLES', 'Layer', 'BaseLayer', 'BuiltLayer']


@six.add_metaclass(ABCMeta)
class LayersConfig(object):

    tf_dtype = tf.float32  # TensorFlow DType
    set_keep = {}  # A dictionary for holding tf.placeholders

    @abstractmethod
    def __init__(self):
        pass


TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


class BaseLayer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    name : str or None
        A unique layer name.

    Methods
    ---------
    print_weights(details=True, session=None)
        Print all parameters of this network.
    print_layers()
        Print all outputs of all layers of this network.
    count_params()
        Return the number of parameters of this network.
    get_all_params()
        Return the parameters in a list of array.

    Examples
    ---------
    - Define model

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100])
    >>> n = tl.layers.InputLayer(x, name='in')
    >>> n = tl.layers.DenseLayer(n, 80, name='d1')
    >>> n = tl.layers.DenseLayer(n, 80, name='d2')

    - Get information

    >>> print(n)
    Last layer is: DenseLayer (d2) [None, 80]
    >>> n.print_layers()
    [TL]   layer   0: d1/Identity:0        (?, 80)            float32
    [TL]   layer   1: d2/Identity:0        (?, 80)            float32
    >>> n.print_weights(False)
    [TL]   param   0: d1/W:0               (100, 80)          float32_ref
    [TL]   param   1: d1/b:0               (80,)              float32_ref
    [TL]   param   2: d2/W:0               (80, 80)           float32_ref
    [TL]   param   3: d2/b:0               (80,)              float32_ref
    [TL]   num of params: 14560
    >>> n.count_params()
    14560

    - Slicing the outputs

    >>> n2 = n[:, :30]
    >>> print(n2)
    Last layer is: Layer (d2) [None, 30]

    - Iterating the outputs

    >>> for l in n:
    >>>    print(l)
    Tensor("d1/Identity:0", shape=(?, 80), dtype=float32)
    Tensor("d2/Identity:0", shape=(?, 80), dtype=float32)

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        return "  Last layer is: %s (%s) %s" % (
            self.__class__.__name__, self.name, self._temp_data['outputs'].get_shape().as_list()
        )

    def __getitem__(self, key):

        net_new = Layer()

        net_new.name = self.name + '_indexing'
        '''
        net_new.inputs = self._temp_data['inputs']
        net_new.outputs = self._temp_data['outputs'][key]

        net_new._add_layers(self.all_layers[:-1])
        net_new._add_layers(net_new.outputs)

        net_new._add_params(self.all_weights)
        net_new._add_graphs(self.all_graphs)
        net_new._add_dropout_layers(self.all_drop)
        '''

        return net_new

    def __setitem__(self, key, item):
        raise TypeError("The Layer API does not allow to use the method: `__setitem__`")

    def __delitem__(self, key):
        raise TypeError("The Layer API does not allow to use the method: `__delitem__`")

    def __iter__(self):
        for x in self.all_layers:
            yield x

    def __len__(self):
        return len(self.all_layers)


class Layer(BaseLayer):

    __metaclass__ = ABCMeta

    @abstractmethod
    @layer_autoregister
    @overwrite_layername_in_network
    def __init__(self, *args, **kwargs):

        super(Layer, self).__init__(*args, **kwargs)

        self._temp_data = None
        self._last_built_layer = None

        self.prev_layer = None

        for key in kwargs.keys():
            setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        if hasattr(self, "act"):
            self.act = self.act if self.act not in [None, tf.identity] else None

    # =============================================== #
    #                  PUBLIC METHODS                 #
    # =============================================== #

    def __str__(self):
        return self._str()

    def __call__(self, prev_layer=None, is_train=True):
        """
        prev_layer : :class:`Layer`
            Previous layer.
        is_train: boolean (default: True)
            Set the TF Variable in training mode and may impact the behaviour of the layer.
        """

        build_layer = False

        ##############################
        #        Sanity Check        #
        ##############################

        if prev_layer is None and self.__class__.__name__ not in tl.layers.inputs.__all__:
            raise ValueError("No previous_layer has been given to the layer `%s`" % self.name)

        ##############################
        #     Manual Layer Build     #
        ##############################

        elif isinstance(prev_layer, tl.layers.Layer):  # Manual Layer Build Mode
            self.prev_layer = prev_layer.name

        ##############################
        #         Layer Build        #
        ##############################

        elif isinstance(prev_layer, tl.layers.BuiltLayer):
            build_layer = True

        ##############################
        #        Input Layers        #
        ##############################

        elif self.__class__.__name__ in tl.layers.inputs.__all__ and (
            isinstance(prev_layer, tf.Tensor) or
            (isinstance(prev_layer,
                        (tuple, list)) and all(isinstance(x, tf.Tensor) for x in prev_layer))
        ):
            build_layer = True

        ##############################
        #       Input is a List      #
        ##############################

        elif isinstance(prev_layer, (list, tuple)):

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #    List of BuiltLayers
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if all(isinstance(layer, tl.layers.BuiltLayer) for layer in prev_layer):

                if any(not hasattr(layer, "outputs") or layer.outputs is None for layer in prev_layer):
                    raise ValueError("A `BuiltLayer` in the layer's inputs contains no output or is None")

                build_layer = True

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #         List of Layers
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            elif all(isinstance(layer, tl.layers.Layer) for layer in prev_layer):
                self.prev_layer = [layer.name for layer in prev_layer]

            else:
                if isinstance(prev_layer, (tuple, list)):
                    additional_details = " - Contents: %s" % [type(x) for x in prev_layer]
                else:
                    additional_details = ""

                raise ValueError(
                    "Not all layers given in `prev_layer` are either `BuiltLayer or `Layer` instance.\n"
                    "Received `prev_layer` - Type: %s%s" % (type(prev_layer), additional_details)
                )

        ##############################
        #   Input is a smthg else    #
        ##############################

        else:
            self.prev_layer = prev_layer

        if build_layer:

            self.parse_inputs(prev_layer, is_train)

            self.build()

            return self._create_built_layer()

        else:
            return self

    # =============================================== #
    #                 PRIVATE METHODS                 #
    # =============================================== #

    @abstractmethod
    @private_method
    def build(self):
        raise NotImplementedError("This method should be overwritten by the derived class")

    @private_method
    def parse_inputs(self, prev_layer, is_train):

        self._temp_data = {
            'inputs': None,
            'unprocessed_inputs': prev_layer,
            'outputs': None,
            'local_weights': list(),
            'local_drop': dict(),
            'is_train': is_train,
        }

        if isinstance(prev_layer, BuiltLayer):
            # 1. for normal layer have only 1 input i.e. DenseLayer
            # Hint : list(), dict() is pass by value (shallow), without them,
            # it is pass by reference.

            self._temp_data['inputs'] = prev_layer.outputs

            #
            #
            # self._add_dropout_layers(prev_layer.all_drop)
            # self._add_graphs(prev_layer.all_graphs)

        elif isinstance(prev_layer, (list, tuple)):
            # 2. for layer have multiply inputs i.e. ConcatLayer

            if self.__class__.__name__ in tl.layers.inputs.__all__:
                self._temp_data['inputs'] = prev_layer
            else:
                self._temp_data['inputs'] = [layer.outputs for layer in prev_layer]

            # self._add_dropout_layers(sum([list(l.all_drop.items()) for l in prev_layer], []))
            # self._add_graphs(sum([l.all_graphs for l in prev_layer], []))

        elif isinstance(prev_layer, tf.Tensor) or isinstance(prev_layer, tf.Variable):  # placeholders
            if self.__class__.__name__ not in tl.layers.inputs.__all__:
                raise RuntimeError("Please use `tl.layers.InputLayer` to convert Tensor/Placeholder to a TL layer")

            self._temp_data['inputs'] = prev_layer
            '''
            self._add_graphs(
                (
                    self._temp_data['inputs'].name,  # .split(':')[0],
                    {
                        'shape': self._temp_data['inputs'].get_shape().as_list(),
                        'dtype': self._temp_data['inputs'].dtype.name,
                        'class': 'placeholder',
                        'prev_layer': None
                    }
                )
            )
            '''

        elif prev_layer is not None:

            # 4. tl.models
            #
            #
            # self._add_dropout_layers(prev_layer.all_drop)
            # self._add_graphs(prev_layer.all_graphs)

            if hasattr(prev_layer, "outputs"):
                self._temp_data['inputs'] = prev_layer.outputs
        '''
        # TL Graph
        if isinstance(prev_layer, list):  # e.g. ConcatLayer, ElementwiseLayer have multiply previous layers
            _list = []

            for layer in prev_layer:
                _list.append(layer.name)

            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': _list})

        elif prev_layer is None:
            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': None})

        else:  # normal layers e.g. Conv2d
            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': prev_layer.name})

        # if act:  # convert activation from function to string
        #     try:
        #         act = act.__name__
        #     except:
        #         pass
        #     self.graph.update({'act': act})
        # print(self.layer_args)

        self.graph.update(self.layer_args)

        self._add_graphs((self.name, self.graph))
        '''

    @private_method
    def _check_list_input(self, layer_list):
        if not isinstance(layer_list, (tuple, list)):
            raise ValueError('`layer_list` should be of type `list` or `tuple`')

        elif not all(isinstance(layer, BuiltLayer) for layer in layer_list):
            raise ValueError("`layer_list` should be a list of `BuiltLayer`")
        else:
            return layer_list

    @private_method
    def _str(self, additional_str=None):

        if additional_str is None or len(additional_str) == 0:
            additional_str = []

        if hasattr(self, 'act'):

            try:
                additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
            except AttributeError:
                pass

        try:
            additional_str.append("output shape: %s" % self._temp_data['outputs'].shape)
        except AttributeError:
            pass

        if len(additional_str) > 0:
            additional_str = ", ".join(additional_str)

        else:
            additional_str = None

        _str = "%s: " % self.__class__.__name__

        if self._temp_data:
            _str += "`%s`" % self.name

            if additional_str is not None:
                _str += " - %s" % additional_str
        else:
            _str += "setup process not finished"

        return _str

    @private_method
    def _skipped_layer_str(self):
        return "  -> [Not Training] - `%s` is skipped. Output shape: %s" % (
            self.__class__.__name__, self._temp_data['outputs'].shape
        )

    @private_method
    def _apply_activation(self, logits, **kwargs):
        if not kwargs:
            kwargs = {}

        if hasattr(self, "act"):
            return self.act(logits, **kwargs) if self.act is not None else logits
        else:
            return logits

    @private_method
    def _create_built_layer(self):
        kwargs = {
            key: val
            for key, val in self._temp_data.items()
            if key not in ['inputs', 'unprocessed_inputs', 'outputs', 'local_weights', 'local_drop', 'is_train']
        }

        self._last_built_layer = type("Built_" + self.__class__.__name__, (BuiltLayer, ), {})(
            layers_to_build=self,
            inputs=self._temp_data['inputs'],
            outputs=self._temp_data['outputs'],
            local_weights=self._temp_data['local_weights'],
            local_drop=self._temp_data['local_drop'],
            is_train=self._temp_data['is_train'],
            **kwargs
        )

        return self._last_built_layer

    @private_method
    def _argument_dict_checkup(self, args):

        if not isinstance(args, dict) and args is not None:
            _err = "One of the argument given to `%s` should be formatted as a dictionary" % self.__class__.__name__
            raise AssertionError(_err)

        return args if args is not None else {}

    # =============================================== #
    #                PROTECTED METHODS                #
    # =============================================== #

    @protected_method
    def _get_tf_variable(
        self,
        name,
        dtype=None,
        shape=None,
        initializer=None,
        regularizer=None,
        trainable=True,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        custom_getter=None,
        constraint=None
    ):
        if dtype is None:
            dtype = self._temp_data['inputs'].dtype

        w = tf.get_variable(
            name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=partitioner,
            validate_shape=validate_shape,
            use_resource=use_resource,
            custom_getter=custom_getter,
            constraint=constraint
        )

        self._add_local_weights(w)

        return w

    @protected_method
    def _add_local_weights(self, weights):

        if isinstance(weights, list):
            for param in weights:
                if param not in self._temp_data['local_weights']:
                    self._temp_data['local_weights'].append(param)

        elif weights not in self._temp_data['local_weights']:
            self._temp_data['local_weights'].append(weights)

    @protected_method
    def _add_local_drop_plh(self, drop_plh, keep_prob):

        if drop_plh not in self._temp_data['local_drop'].keys():
            self._temp_data['local_drop'].update({drop_plh: keep_prob})

    @protected_method
    def _get_init_args(self, skip=4):
        """Get all arguments of current layer for saving the graph."""
        stack = inspect.stack()

        if len(stack) < skip + 1:
            raise ValueError("The length of the inspection stack is shorter than the requested start position.")

        args, _, _, values = inspect.getargvalues(stack[skip][0])

        params = {}

        for arg in args:

            # some args dont need to be saved into the graph. e.g. the input placeholder
            if values[arg] is None or arg in ['self', 'prev_layer', 'inputs']:
                continue

            else:
                val = values[arg]

                # change function (e.g. act) into dictionary of module path and function name
                if inspect.isfunction(val):
                    params[arg] = {"module_path": val.__module__, "func_name": val.__name__}

                # ignore more args e.g. TF class
                elif arg.endswith('init'):
                    continue

                # for other data type, save them directly
                else:
                    params[arg] = val

        return params

    # =============================================== #
    #              TO BE REMOVED/MOVED                #
    # =============================================== #

    @protected_method
    def _add_layers(self, layers):
        tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, '_add_layers'))
        pass
        '''
        if isinstance(layers, list):
            try:  # list of class Layer
                new_layers = [layer.outputs for layer in layers]
                self.all_layers.extend(list(new_layers))

            except AttributeError:  # list of tf.Tensor
                self.all_layers.extend(list(layers))

        else:
            self.all_layers.append(layers)

        self.all_layers = list_remove_repeat(self.all_layers)
        '''

    @protected_method
    def _add_params(self, params):
        tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, '_add_params'))
        pass
        '''
        if isinstance(params, list):
            for param in params:
                if param not in self.all_weights:
                    self.all_weights.append(param)

        elif params not in self.all_weights:
            self.all_weights.append(params)
        '''

    @protected_method
    def _add_graphs(self, graphs):
        tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, '_add_graphs'))
        pass
        '''
        if isinstance(graphs, list):
            for graph in graphs:
                if graph not in self.all_graphs:
                    self.all_graphs.append(graph)

        elif graphs not in self.all_graphs:
            self.all_graphs.append(graphs)
        '''

    @protected_method
    def _add_dropout_layers(self, drop_layers):
        tl.logging.fatal(
            "THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, '_add_dropout_layers')
        )
        pass
        '''
        if isinstance(drop_layers, dict) or isinstance(drop_layers, list):
            self.all_drop.update(dict(drop_layers))

        elif isinstance(drop_layers, tuple):
            self.all_drop.update(list(drop_layers))

        else:
            raise ValueError()
        '''

    # should be renamed `get_all_weights`
    def get_all_params(self, session=None):
        """Return the parameters in a list of array."""
        tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, 'get_all_params'))
        pass
        '''
        _params = []
        for p in self.all_weights:
            if session is None:
                _params.append(p.eval())
            else:
                _params.append(session.run(p))
        return _params
        '''

    # should be renamed `print_network`
    def print_layers(self):
        """Print all info of layers in the network."""
        tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, 'print_layers'))
        pass
        '''
        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info(
                "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            )
        '''

    # def __getstate__(self): # pickle save
    #     return {'version': 0.1,
    #             # 'outputs': self._temp_data['outputs'],
    #             }
    #
    # def __setstate__(self, state): # pickle restore
    #     self.outputs = state['outputs']


class BuiltLayer(object):

    def __init__(self, layers_to_build, inputs, outputs, local_weights, local_drop, is_train, **kwargs):

        self.hyperparameters = dict()

        for key, value in layers_to_build.__dict__.items():

            # Do not record these arguments

            # "prev_layer", "all_params", "all_layers", "all_drop", "all_graphs", "graph", "is_setup",
            # "inputs", "outputs", "_local_drop", "_local_weights",

            if key in ["_last_built_layer", '_temp_data']:
                continue

            # setattr(self, key, value)
            self.hyperparameters[key] = value

        self._str_ = str(layers_to_build)

        self.inputs = inputs
        self.outputs = outputs

        self.local_weights = local_weights
        self.local_drop = local_drop

        self.is_train = is_train

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.name = self.hyperparameters["name"]

        logging.info(str(self))

    def __str__(self):
        return self._str_

    def __setattr__(self, key, value):

        previous_frame = inspect.currentframe().f_back
        _, _, function_name, _, _ = inspect.getframeinfo(previous_frame)

        if function_name == "__init__":
            super(BuiltLayer, self).__setattr__(key, value)
        else:
            raise RuntimeError(
                "A Tensorlayer `{}` is not supposed to be modified.\n"
                "An attempt to modify the attribute: `{}` has been detected.".format(self.__class__.__name__, key)
            )

    # =============================================== #
    #                  PUBLIC METHODS                 #
    # =============================================== #

    def count_local_weights(self):

        """Returns the number of parameters in the network."""
        n_params = 0

        for _i, p in enumerate(self.local_weights):

            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():

                try:
                    s = int(s)
                except (TypeError, ValueError):
                    s = 1

                if s:
                    n = n * s

            n_params = n_params + n

        return n_params

    def count_weights(self):
        """Returns the number of weights in the network."""
        n_params = 0

        for _i, p in enumerate(self.local_weights):

            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():

                try:
                    s = int(s)
                except (TypeError, ValueError):
                    s = 1

                if s:
                    n = n * s

            n_params = n_params + n

        return n_params

    # should be renamed `print_weights`
    def print_weights(self, details=True, session=None):
        """Print all info of weights in the network"""

        for i, p in enumerate(self.local_weights):
            if details:
                try:
                    val = p.eval(session=session)
                    logging.info(
                        "  weight {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".
                        format(i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std())
                    )
                except Exception as e:
                    logging.info(str(e))
                    raise Exception(
                        "Hint: print_weights can only be used after the session have been initialized "
                        "or use network.print_weights(False)."
                    )
            else:
                logging.info("  weight {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of weights: %d" % self.count_params())

    # =============================================== #
    #              TO BE REMOVED/MOVED                #
    # =============================================== #

    @deprecated(
        end_support_version="2.1.0", instructions="`count_params` has been deprecated in favor of `count_weights`"
    )  # TODO: remove this line before releasing TL 2.1.0
    def count_params(self):
        """Returns the number of parameters in the network"""
        return self.count_weights()

    @deprecated(
        end_support_version="2.1.0", instructions="`count_params` has been deprecated in favor of `count_weights`"
    )  # TODO: remove this line before releasing TL 2.1.0
    def print_params(self):
        """Returns the number of parameters in the network"""
        return self.count_weights()

    # should be renamed `print_network`
    def print_layers(self):
        """Print all info of layers in the network."""
        # tl.logging.fatal("THIS FUNCTION WILL BE REMOVED SOON: %s.%s()" % (self.__class__.__name__, 'print_layers'))
        # pass

        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info(
                "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            )
