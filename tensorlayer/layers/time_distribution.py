# -*- coding: utf-8 -*-

from .core import *


class TimeDistributedLayer(Layer):
    """
    The :class:`TimeDistributedLayer` class that applies a function to every timestep of the input tensor.
    For example, if use :class:`DenseLayer` as the `layer_class`, we input (batch_size, length, dim) and
    output (batch_size , length, new_dim).

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer with output size of (batch_size, length, dim).
    layer_class : a :class:`Layer` class
        The layer class name.
    args : dictionary
        The arguments for the ``layer_class``.
    name : str
        A unique layer name.

    Examples
    --------
    >>> batch_size = 32
    >>> timestep = 20
    >>> input_dim = 100
    >>> x = tf.placeholder(dtype=tf.float32, shape=[batch_size, timestep, input_dim], name="encode_seqs")
    >>> net = InputLayer(x, name='input')
    >>> net = TimeDistributedLayer(net, layer_class=DenseLayer, args={'n_units':50, 'name':'dense'}, name='time_dense')
    ... [TL] InputLayer  input: (32, 20, 100)
    ... [TL] TimeDistributedLayer time_dense: layer_class:DenseLayer
    >>> print(net.outputs._shape)
    ... (32, 20, 50)
    >>> net.print_params(False)
    ... param   0: (100, 50)          time_dense/dense/W:0
    ... param   1: (50,)              time_dense/dense/b:0
    ... num of params: 5050

    """

    def __init__(
            self,
            layer,
            layer_class=None,
            args=None,
            name='time_distributed',
    ):
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise TypeError("'args' must be a dict.")

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("TimeDistributedLayer %s: layer_class:%s args:%s" % (self.name, layer_class.__name__, args))

        if not isinstance(self.inputs, tf.Tensor):
            self.inputs = tf.transpose(tf.stack(self.inputs), [1, 0, 2])

        input_shape = self.inputs.get_shape()

        timestep = input_shape[1]
        x = tf.unstack(self.inputs, axis=1)

        for i in range(0, timestep):
            with tf.variable_scope(name, reuse=(set_keep['name_reuse'] if i == 0 else True)) as vs:
                set_name_reuse((set_keep['name_reuse'] if i == 0 else True))
                net = layer_class(InputLayer(x[i], name=args['name'] + str(i)), **args)
                # net = layer_class(InputLayer(x[i], name="input_"+args['name']), **args)
                x[i] = net.outputs
                variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.outputs = tf.stack(x, axis=1, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
