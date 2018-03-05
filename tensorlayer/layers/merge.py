# -*- coding: utf-8 -*-

from .core import *


class ConcatLayer(Layer):
    """A layer that concats multiple tensors according to given axis..

    Parameters
    ----------
    layers : list of :class:`Layer`
        List of layers to concatenate.
    concat_dim : int
        The dimension to concatenate.
    name : str
        A unique layer name.

    Examples
    ----------
    >>> sess = tf.InteractiveSession()
    >>> x = tf.placeholder(tf.float32, shape=[None, 784])
    >>> inputs = tl.layers.InputLayer(x, name='input_layer')
    >>> net1 = tl.layers.DenseLayer(inputs, 800, act=tf.nn.relu, name='relu1_1')
    >>> net2 = tl.layers.DenseLayer(inputs, 300, act=tf.nn.relu, name='relu2_1')
    >>> net = tl.layers.ConcatLayer([net1, net2], 1, name ='concat_layer')
    ...   InputLayer input_layer (?, 784)
    ...   DenseLayer relu1_1: 800, relu
    ...   DenseLayer relu2_1: 300, relu
    ...   ConcatLayer concat_layer, 1100
    >>> tl.layers.initialize_global_variables(sess)
    >>> net.print_params()
    ...     param 0: (784, 800) (mean: 0.000021, median: -0.000020 std: 0.035525)
    ...     param 1: (800,)     (mean: 0.000000, median: 0.000000  std: 0.000000)
    ...     param 2: (784, 300) (mean: 0.000000, median: -0.000048 std: 0.042947)
    ...     param 3: (300,)     (mean: 0.000000, median: 0.000000  std: 0.000000)
    ...     num of params: 863500
    >>> net.print_layers()
    ...     layer 0: ("Relu:0", shape=(?, 800), dtype=float32)
    ...     layer 1: Tensor("Relu_1:0", shape=(?, 300), dtype=float32)

    """

    def __init__(
            self,
            layers,
            concat_dim=1,
            name='concat_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = []
        for l in layers:
            self.inputs.append(l.outputs)
        try:  # TF1.0
            self.outputs = tf.concat(self.inputs, concat_dim, name=name)
        except Exception:  # TF0.12
            self.outputs = tf.concat(concat_dim, self.inputs, name=name)

        logging.info("ConcatLayer %s: axis: %d" % (self.name, concat_dim))

        self.all_layers = list(layers[0].all_layers)
        self.all_params = list(layers[0].all_params)
        self.all_drop = dict(layers[0].all_drop)

        for i in range(1, len(layers)):
            self.all_layers.extend(list(layers[i].all_layers))
            self.all_params.extend(list(layers[i].all_params))
            self.all_drop.update(dict(layers[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


class ElementwiseLayer(Layer):
    """A layer that combines multiple :class:`Layer` that have the same output shapes
    according to an element-wise operation.

    Parameters
    ----------
    layers : list of :class:`Layer`
        The list of layers to combine.
    combine_fn : a TensorFlow element-wise combine function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`__ .
    act : activation function
        The activation function of this layer.
    name : str
        A unique layer name.

    Examples
    --------
    AND Logic

    >>> net_0 = tl.layers.DenseLayer(net_0, n_units=500,
    ...                        act = tf.nn.relu, name='net_0')
    >>> net_1 = tl.layers.DenseLayer(net_1, n_units=500,
    ...                        act = tf.nn.relu, name='net_1')
    >>> net_com = tl.layers.ElementwiseLayer(layer = [net_0, net_1],
    ...                         combine_fn = tf.minimum,
    ...                         name = 'combine_layer')

    """

    def __init__(
            self,
            layers,
            combine_fn=tf.minimum,
            act=None,
            name='elementwise_layer',
    ):
        Layer.__init__(self, name=name)

        logging.info("ElementwiseLayer %s: size:%s fn:%s" % (self.name, layers[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layers[0].outputs

        for l in layers[1:]:
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)

        if act:
            self.outputs = act(self.outputs)

        self.all_layers = list(layers[0].all_layers)
        self.all_params = list(layers[0].all_params)
        self.all_drop = dict(layers[0].all_drop)

        for i in range(1, len(layers)):
            self.all_layers.extend(list(layers[i].all_layers))
            self.all_params.extend(list(layers[i].all_params))
            self.all_drop.update(dict(layers[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)
        # self.all_drop = list_remove_repeat(self.all_drop)
