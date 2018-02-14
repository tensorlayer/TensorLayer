# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *


class SlimNetsLayer(Layer):
    """
    The :class:`SlimNetsLayer` class can be used to merge all TF-Slim nets into
    TensorLayer. Models can be found in `slim-model <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`_,
    see Inception V3 example on `Github <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`_.


    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    slim_layer : a slim network function
        The network you want to stack onto, end with ``return net, end_points``.
    slim_args : dictionary
        The arguments for the slim model.
    name : a string or None
        An optional name to attach to this layer.

    Notes
    -----
    The due to TF-Slim stores the layers as dictionary, the ``all_layers`` in this
    network is not in order ! Fortunately, the ``all_params`` are in order.
    """

    def __init__(
            self,
            layer=None,
            slim_layer=None,
            slim_args={},
            name='tfslim_layer',
    ):
        Layer.__init__(self, name=name)
        assert slim_layer is not None
        assert slim_args is not None
        self.inputs = layer.outputs
        print("  [TL] SlimNetsLayer %s: %s" % (self.name, slim_layer.__name__))

        # with tf.variable_scope(name) as vs:
        #     net, end_points = slim_layer(self.inputs, **slim_args)
        #     slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        net, end_points = slim_layer(self.inputs, **slim_args)

        slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=name)
        if slim_variables == []:
            print(
                "No variables found under %s : the name of SlimNetsLayer should be matched with the begining of the ckpt file, see tutorial_inceptionV3_tfslim.py for more details"
                % name)

        self.outputs = net

        slim_layers = []
        for v in end_points.values():
            # tf.contrib.layers.summaries.summarize_activation(v)
            slim_layers.append(v)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend(slim_layers)
        self.all_params.extend(slim_variables)
