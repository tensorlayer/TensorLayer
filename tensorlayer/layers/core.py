#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect
import six

from abc import ABCMeta, abstractmethod

import numpy as np

import tensorflow as tf

from tensorlayer.layers.utils import list_remove_repeat, get_variable_with_initializer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import protected_method
from tensorlayer.decorators import private_method

__all__ = [
    # 'LayersConfig',  # TODO : remove this??
    # 'TF_GRAPHKEYS_VARIABLES',  # TODO : remove this??
    'Layer',
]

_global_layer_name_dict = {}  # TODO: better implementation?

# @six.add_metaclass(ABCMeta)
# class LayersConfig(object):
#
#     tf_dtype = tf.float32  # TensorFlow DType
#     set_keep = {}  # A dictionary for holding tf.placeholders
#
#     @abstractmethod
#     def __init__(self):
#         pass

# TF_GRAPHKEYS_VARIABLES = tf.compat.v1.GraphKeys.GLOBAL_VARIABLES


class Layer(object):
    #FIXME: documentation update needed
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
    check this https://github.com/luomai/tensorlayer2-design/issues/7
    # print_weights(details=True, session=None)
    #     Print all parameters of this network.
    # print_layers()
    #     Print all outputs of all layers of this network.
    # count_weights()
    #     Return the number of parameters of this network.
    # get_all_weights()
    #     Return the parameters in a list of array.

    Examples
    ---------
    - Define model

    >>> import tensorflow as tf
    >>> import tensorlayer as tl
    >>> x = tf.placeholder("float32", [None, 100])      # TODO: rewrite
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
    [TL]   num of weights: 14560
    >>> n.count_weights()
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
    def __init__(self, name=None, act=None, *args, **kwargs):
        # Layer constants

        for key in kwargs.keys():
            setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        self.act = act if act not in [None, tf.identity] else None

        ## Hao Dong: automatically add layer type as the prefix of the layers
        global _global_layer_name_dict
        if name is None:
            prefix = self.__class__.__name__.lower()
            if _global_layer_name_dict.get(prefix) is not None:
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
            else:
                _global_layer_name_dict[prefix] = 0
                name = prefix

        # FIXME: double check needed: the scope name may be deprecated in TF2
        # scope_name = tf.get_variable_scope().name
        # self.name = scope_name + '/' + name if scope_name else name
        self.name = name

        # Layer input outputs
        self.inputs = None
        self.outputs = None  # TODO: not accessible to eager mode but accessible to graph mode

        # self._inputs_shape = None
        # self._outputs_shape = None

        self._input_layer = None

        # TODO: need to update
        # self.all_layers = list()  # we change layers --> outputs ?
        # self.all_weights = list()  # we change weights --> weights ?
        # self.all_drop = dict()    # remove all_drop

        # Layer building state
        self._built = False

        # Layer weight state
        self._weights = None

    @property
    def _inputs_shape(self):  # TODO, if self.outputs is a list ???
        return self._input_layer._outputs_shape

    @property
    def _outputs_shape(self):  # TODO, if self.outputs is a list ???
        return self.outputs.get_shape().as_list()

    @property
    def weights(self):
        return self._weights

    def __call__(self, prev_layer):

        if self._built:
            raise Exception("The layer has been built before.")

        if isinstance(prev_layer, Layer):
            # 1. for normal layer have only 1 input i.e. DenseLayer
            # Hint : list(), dict() is pass by value (shallow), without them,
            # it is pass by reference.

            self.inputs = prev_layer.outputs
            self._input_layer = prev_layer
            # self._inputs_shape = self._input_layer._outputs_shape

            self._weights = list()

            self.build(self._inputs_shape)
            self.outputs = self.forward(self.inputs, is_train=False)
            # self._outputs_shape = self.outputs.get_shape().as_list()

            self._built = True

            # TODO: need update
            # self._add_layers(prev_layer.all_layers)
            # self._add_weights(self._weights)
            # self._add_weights(prev_layer.all_weights)
            # self._add_dropout_layers(prev_layer.all_drop)

        else:
            # FIXME: not sure yet how to handle other cases
            # FIXME: if the prev_layer is a list, corresponding code in Model() needs update
            '''
            elif isinstance(prev_layer, list):
                # 2. for layer have multiply inputs i.e. ConcatLayer

                self.inputs = [layer.outputs for layer in prev_layer]

                self._add_layers(sum([l.all_layers for l in prev_layer], []))
                self._add_weights(sum([l.all_weights for l in prev_layer], []))
                self._add_dropout_layers(sum([list(l.all_drop.items()) for l in prev_layer], []))

            elif isinstance(prev_layer, tf.Tensor) or isinstance(prev_layer, tf.Variable):  # placeholders
                if self.__class__.__name__ not in ['InputLayer', 'OneHotInputLayer', 'Word2vecEmbeddingInputlayer',
                                                   'EmbeddingInputlayer', 'AverageEmbeddingInputlayer']:
                    raise RuntimeError("Please use `tl.layers.InputLayer` to convert Tensor/Placeholder to a TL layer")

                self.inputs = prev_layer

            elif prev_layer is not None:
                # 4. tl.models
                self._add_layers(prev_layer.all_layers)
                self._add_weights(prev_layer.all_weights)
                self._add_dropout_layers(prev_layer.all_drop)

                if hasattr(prev_layer, "outputs"):
                    self.inputs = prev_layer.outputs
            '''

        return self

    # def _get_weights(self, scope_name, var_name, shape, init=np.random.normal, init_args=None):
    def _get_weights(self, var_name, shape, init=tf.initializers.random_normal, init_args=None):
        weight = get_variable_with_initializer(
            scope_name=self.name, var_name=var_name, shape=shape, init=init, init_args=init_args
        )
        self._weights.append(weight)  # Add into the weight collection
        # self.__setattr__(var_name, weight) # FIXME: prefer to remove this line, the weights should be manually defined as members of the Layer
        return weight

    @abstractmethod
    def build(self, inputs_shape):
        # FIXME: documentation needed
        """
        An abstract method which should be overwritten in derived classes to define all necessary weights of the layer.

        :param inputs_shape: tuple
        :return: void
        """
        raise Exception("The build_weights method must be implemented by inherited class")

    @abstractmethod
    def forward(self, inputs, is_train):
        # FIXME: documentation needed
        """
        An abstract method which should be overwritten in derived classes to define forward feeding operations of the layer.

        :param inputs: Tensor
        :param is_train: boolean, True for training and False for testing
        :return: Tensor
        """
        raise Exception("The forward method must be implemented by inherited class")

    def print_weights(self, details=False, session=None):
        """Print all information of weights in the model. """
        for i, p in enumerate(self.all_weights):
            if details:
                try:
                    val = p.eval(session=session)
                    logging.info(
                        "  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".
                        format(i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std())
                    )
                except Exception as e:
                    logging.info(str(e))
                    raise Exception(
                        "Hint: print weights details after tl.layers.initialize_global_variables(sess) "
                        "or use network.print_weights(False)."
                    )
            else:
                logging.info("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        logging.info("  num of weights: %d" % self.count_weights())

    # TODO: deprecated if no all_layers
    def print_layers(self):
        """Print all info of layers in the network."""
        for i, layer in enumerate(self.all_layers):
            # logging.info("  layer %d: %s" % (i, str(layer)))
            logging.info(
                "  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name)
            )

    # TODO: need to rewrite
    def count_weights(self):
        """Returns the number of parameters in the network."""
        n_weights = 0
        for _i, p in enumerate(self.all_weights):
            n = 1
            # for s in p.eval().shape:
            for s in p.get_shape():
                try:
                    s = int(s)
                except Exception:
                    s = 1
                if s:
                    n = n * s
            n_weights = n_weights + n
        return n_weights

    @property
    def n_weights():
        return count_weights()

    # TODO: need to rewrite
    def get_all_weights(self, sess=None):
        """Return the weights in a list of array."""
        _weights = []
        for p in self.all_weights:
            if sess is None:
                _weights.append(p.eval())
            else:
                _weights.append(sess.run(p))
        return _weights

    def __str__(self):
        if self.outputs is not None:
            _outputs_shape = [tuple(['batch_size'] + o.shape.as_list()) for o in self.outputs]
            if len(_outputs_shape) == 1:
                _outputs_shape = _outputs_shape[0]
        else:
            _outputs_shape = "unknown for unbuilt layer"
        return "  {} ({}) outputs_shape: {}".format(self.__class__.__name__, self.name, _outputs_shape)
        # self._outputs_shape)#outputs.get_shape().as_list())

    # def __getitem__(self, key):
    #
    #     net_new = Layer(prev_layer=None, name=self.name)
    #
    #     net_new.name = self.name + '_indexing'
    #     net_new.inputs = self.inputs
    #     net_new.outputs = self.outputs[key]
    #
    #     net_new._add_layers(self.all_layers[:-1])
    #     net_new._add_layers(net_new.outputs)
    #
    #     net_new._add_weights(self.all_weights)
    #     # net_new._add_dropout_layers(self.all_drop)
    #
    #     return net_new

    def __setitem__(self, key, item):
        raise TypeError("The Layer API does not allow to use the method: `__setitem__`")

    def __delitem__(self, key):
        raise TypeError("The Layer API does not allow to use the method: `__delitem__`")

    def __iter__(self):
        for x in self.all_layers:  # FIXME: it is good for eager mode?
            yield x

    def __len__(self):
        return len(self.all_layers)

    @protected_method
    def _get_init_args(self, skip=4):
        """Get all arguments of current layer for the configuration information."""
        stack = inspect.stack()

        if len(stack) < skip + 1:
            raise ValueError("The length of the inspection stack is shorter than the requested start position.")

        args, _, _, values = inspect.getargvalues(stack[skip][0])

        weights = {}

        for arg in args:

            # some args dont need to be saved into the graph. e.g. the input placeholder
            if values[arg] is not None and arg not in ['self', 'prev_layer', 'inputs']:

                val = values[arg]

                # change function (e.g. act) into dictionary of module path and function name
                if inspect.isfunction(val):
                    weights[arg] = {"module_path": val.__module__, "func_name": val.__name__}
                # ignore more args e.g. TF class
                elif arg.endswith('init'):
                    continue
                # for other data type, save them directly
                else:
                    weights[arg] = val

        return weights

    # # todo: deprecated if no all_layer
    # @protected_method
    # def _add_layers(self, layers):
    #     if isinstance(layers, list):
    #         try:  # list of class Layer
    #             new_layers = [layer.outputs for layer in layers]
    #             self.all_layers.extend(list(new_layers))
    #
    #         except AttributeError:  # list of tf.Tensor
    #             self.all_layers.extend(list(layers))
    #
    #     else:
    #         self.all_layers.append(layers)
    #
    #     self.all_layers = list_remove_repeat(self.all_layers)

    # # todo: deprecated if no all_weights
    # @protected_method
    # def _add_weights(self, weights):
    #
    #     if isinstance(weights, list):
    #         self.all_weights.extend(list(weights))
    #
    #     else:
    #         self.all_weights.append(weights)
    #
    #     self.all_weights = list_remove_repeat(self.all_weights)

    # @protected_method
    # def _add_dropout_layers(self, drop_layers):
    #     if isinstance(drop_layers, dict) or isinstance(drop_layers, list):
    #         self.all_drop.update(dict(drop_layers))
    #
    #     elif isinstance(drop_layers, tuple):
    #         self.all_drop.update(list(drop_layers))
    #
    #     else:
    #         raise ValueError()

    # FIXME: may not be necessary ???  Hao: I think it is not necessary..
    @private_method
    def _apply_activation(self, logits, **kwargs):
        if not kwargs:
            kwargs = {}
        return self.act(logits, **kwargs) if self.act is not None else logits

    # TODO: may need update
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

    ## raise Exceptions for old version codes
    def count_params(self, **kwargs):
        raise Exception("please change count_params --> count_weights")

    def print_params(self, **kwargs):
        raise Exception("please change print_params --> print_weights")

    @property
    def all_params(self):
        raise Exception("please change all_params --> weights")
