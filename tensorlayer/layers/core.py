#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import six

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow as tf

from tensorlayer.layers.utils import list_remove_repeat

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import protected_method
from tensorlayer.decorators import private_method

__all__ = [
    'LayersConfig',
    'TF_GRAPHKEYS_VARIABLES',
    'Layer',
]


@six.add_metaclass(ABCMeta)
class LayersConfig(object):

    tf_dtype = tf.float32  # TensorFlow DType
    set_keep = {}  # A dictionary for holding tf.placeholders

    @abstractmethod
    def __init__(self):
        pass


TF_GRAPHKEYS_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


class Layer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    prev_layer : :class:`Layer` or None
        Previous layer (optional), for adding all properties of previous layer(s) to this layer.
    act : activation function (None by default)
        The activation function of this layer.
    name : str or None
        A unique layer name.

    Methods
    ---------
    print_params(details=True, session=None)
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
    >>> n.print_params(False)
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
    # Added to allow auto-completion

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, act=None, name=None, *args, **kwargs):

        self.inputs = None
        self.outputs = None
        self.graph = {}
        self.all_layers = list()
        self.all_params = list()
        self.all_drop = dict()
        self.all_graphs = list()

        self.layer_args = self._get_init_args(skip=4)

        if name is None:
            raise ValueError('Layer must have a name.')

        for key in kwargs.keys():
            setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        self.act = act if act not in [None, tf.identity] else None

        scope_name = tf.get_variable_scope().name

        self.name = scope_name + '/' + name if scope_name else name

        if isinstance(prev_layer, Layer):
            # 1. for normal layer have only 1 input i.e. DenseLayer
            # Hint : list(), dict() is pass by value (shallow), without them,
            # it is pass by reference.

            self.inputs = prev_layer.outputs

            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)
            self._add_graphs(prev_layer.all_graphs)

        elif isinstance(prev_layer, list):
            # 2. for layer have multiply inputs i.e. ConcatLayer

            self.inputs = [layer.outputs for layer in prev_layer]

            self._add_layers(sum([l.all_layers for l in prev_layer], []))
            self._add_params(sum([l.all_params for l in prev_layer], []))
            self._add_dropout_layers(sum([list(l.all_drop.items()) for l in prev_layer], []))
            self._add_graphs(sum([l.all_graphs for l in prev_layer], []))

        elif isinstance(prev_layer, tf.Tensor) or isinstance(prev_layer, tf.Variable):  # placeholders
            if self.__class__.__name__ not in ['InputLayer', 'OneHotInputLayer', 'Word2vecEmbeddingInputlayer',
                                               'EmbeddingInputlayer', 'AverageEmbeddingInputlayer']:
                raise RuntimeError("Please use `tl.layers.InputLayer` to convert Tensor/Placeholder to a TL layer")

            self.inputs = prev_layer

            self._add_graphs((self.inputs.name, #.split(':')[0],
                {'shape': self.inputs.get_shape().as_list(),
                'dtype': self.inputs.dtype.name, 'class': 'placeholder',
                'prev_layer': None}))

        elif prev_layer is not None:
            # 4. tl.models
            self._add_layers(prev_layer.all_layers)
            self._add_params(prev_layer.all_params)
            self._add_dropout_layers(prev_layer.all_drop)
            self._add_graphs(prev_layer.all_graphs)

            if hasattr(prev_layer, "outputs"):
                self.inputs = prev_layer.outputs

        ## TL Graph
        if isinstance(prev_layer, list):  # e.g. ConcatLayer, ElementwiseLayer have multiply previous layers
            _list = []
            for layer in prev_layer:
                _list.append(layer.name)
            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': _list})
        elif prev_layer is None:  #
            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': None})
        else:  # normal layers e.g. Conv2d
            self.graph.update({'class': self.__class__.__name__.split('.')[-1], 'prev_layer': prev_layer.name})
        # if act:  ## convert activation from function to string
        #     try:
        #         act = act.__name__
        #     except:
        #         pass
        #     self.graph.update({'act': act})
        # print(self.layer_args)
        self.graph.update(self.layer_args)
        # print(self.graph)
        self._add_graphs((self.name, self.graph))

    def print_params(self, details=True, session=None):
        """Print all info of parameters in the network"""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    val = p.eval(session=session)
                    logging.info(
                        "  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                            i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()
                        )
                    )
                except Exception as e:
                    logging.info(str(e))
                    raise Exception(
                        "Hint: print params details after tl.layers.initialize_global_variables(sess) "
                        "or use network.print_params(False)."
                    )
            else:
                logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network"""

        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info(
                "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            )

    def count_params(self):
        """Returns the number of parameters in the network"""
        n_params = 0
        for _i, p in enumerate(self.all_params):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except Exception:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def get_all_params(self, session=None):
        """Return the parameters in a list of array. """
        _params = []
        for p in self.all_params:
            if session is None:
                _params.append(p.eval())
            else:
                _params.append(session.run(p))
        return _params

    def __str__(self):
        return "  Last layer is: %s (%s) %s" % (self.__class__.__name__, self.name, self.outputs.get_shape().as_list())

    def __getitem__(self, key):

        net_new = Layer(prev_layer=None, name=self.name)

        net_new.name = self.name + '_indexing'
        net_new.inputs = self.inputs
        net_new.outputs = self.outputs[key]

        net_new._add_layers(self.all_layers[:-1])
        net_new._add_layers(net_new.outputs)

        net_new._add_params(self.all_params)
        net_new._add_graphs(self.all_graphs)
        net_new._add_dropout_layers(self.all_drop)

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

    @protected_method
    def _get_init_args(self, skip=4):
        """Get all arguments of current layer for saving the graph. """
        stack = inspect.stack()

        if len(stack) < skip + 1:
            raise ValueError("The length of the inspection stack is shorter than the requested start position.")

        args, _, _, values = inspect.getargvalues(stack[skip][0])

        params = {}

        for arg in args:

            ## some args dont need to be saved into the graph. e.g. the input placeholder
            if values[arg] is not None and arg not in ['self', 'prev_layer', 'inputs']:

                val = values[arg]

                ## change function (e.g. act) into dictionary of module path and function name
                if inspect.isfunction(val):
                    params[arg] = {"module_path": val.__module__, "func_name": val.__name__}
                ## ignore more args e.g. TF class
                elif arg.endswith('init'):
                    continue
                ## for other data type, save them directly
                else:
                    params[arg] = val

        return params

    @protected_method
    def _add_layers(self, layers):
        if isinstance(layers, list):
            try:  # list of class Layer
                new_layers = [layer.outputs for layer in layers]
                self.all_layers.extend(list(new_layers))

            except AttributeError:  # list of tf.Tensor
                self.all_layers.extend(list(layers))

        else:
            self.all_layers.append(layers)

        self.all_layers = list_remove_repeat(self.all_layers)

    @protected_method
    def _add_params(self, params):

        if isinstance(params, list):
            self.all_params.extend(list(params))

        else:
            self.all_params.append(params)

        self.all_params = list_remove_repeat(self.all_params)

    @protected_method
    def _add_graphs(self, graphs):

        if isinstance(graphs, list):
            self.all_graphs.extend(list(graphs))

        else:
            self.all_graphs.append(graphs)

        # self.all_graphs = list_remove_repeat(self.all_graphs) # cannot repeat

    @protected_method
    def _add_dropout_layers(self, drop_layers):
        if isinstance(drop_layers, dict) or isinstance(drop_layers, list):
            self.all_drop.update(dict(drop_layers))

        elif isinstance(drop_layers, tuple):
            self.all_drop.update(list(drop_layers))

        else:
            raise ValueError()

    @private_method
    def _apply_activation(self, logits, **kwargs):
        if not kwargs:
            kwargs = {}
        return self.act(logits, **kwargs) if self.act is not None else logits

    @private_method
    def _argument_dict_checkup(self, args):

        if not isinstance(args, dict) and args is not None:
            raise AssertionError(
                "One of the argument given to %s should be formatted as a dictionary" % self.__class__.__name__
            )

        return args if args is not None else {}

    # def __getstate__(self): # pickle save
    #     return {'version': 0.1,
    #             # 'outputs': self.outputs,
    #             }
    #
    # def __setstate__(self, state): # pickle restore
    #     self.outputs = state['outputs']
