#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'SlimNets',
    'Keras',
]


class SlimNets(Layer):
    """A layer that merges TF-Slim models into TensorLayer.

    Models can be found in `slim-model <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`__,
    see Inception V3 example on `Github <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`__.

    Parameters
    ----------
    slim_layer : a slim network function
        The network you want to stack onto, end with ``return net, end_points``.
    slim_args : dictionary
        The arguments for the slim model.
    name : str
        A unique layer name.

    Notes
    -----
    - As TF-Slim stores the layers as dictionary, the ``all_layers`` in this network is not in order ! Fortunately, the ``all_params`` are in order.

    """

    def __init__(
        self,
        slim_layer,
        slim_args=None,
        act=None,
        name='tfslim',
    ):

        if slim_layer is None:
            raise ValueError("slim layer is None")

        self.slim_layer = slim_layer
        self.act = act
        self.name = name

        super(SlimNets, self).__init__(slim_args=slim_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer kind: %s" % self.slim_layer.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        slim_layers = []

        with tf.variable_scope(self.name) as vs:

            _out = self.slim_layer(self._temp_data['inputs'], **self.slim_args)

            if isinstance(_out, tf.Tensor):
                self._temp_data['outputs'] = _out
                slim_layers.append(_out)

            else:
                self._temp_data['outputs'], end_points = _out

                for v in end_points.values():
                    slim_layers.append(v)

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


# @deprecated(
#     end_support_version="2.0.0", instructions="This layer will be removed in TL 2.0.0 in favor of :class:`Lambda`"
# )  # TODO: remove this line before releasing TL 2.0.0
class Keras(Layer):
    """A layer to import Keras layers into TensorLayer.

    Example can be found here `tutorial_keras.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_keras.py>`__.

    Parameters
    ----------
    keras_layer : function
        A tensor in tensor out function for building model.
    keras_args : dictionary
        The arguments for the `keras_layer`.
    name : str
        A unique layer name.

    """

    def __init__(
        self,
        keras_layer,
        keras_args=None,
        act=None,
        name='keras',
    ):

        if keras_layer is None:
            raise ValueError("keras_layer is None")

        self.act = act
        self.name = name
        self.keras_layer = keras_layer(**keras_args)

        if not isinstance(self.keras_layer, tf.keras.layers.Layer):
            raise ValueError("keras_layer is not a Keras Layer but `%s`" % type(self.keras_layer))

        super(Keras, self).__init__(keras_args=keras_args)

        logging.warning("This API will be removed, please use `Lambda` instead.")

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer kind: %s" % self.keras_layer.__name__)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        current_varscope = tf.get_variable_scope()

        with tf.variable_scope(current_varscope.name + "/" + self.name + "/", reuse=current_varscope.reuse):

            self._temp_data['outputs'] = self.keras_layer(self._temp_data['inputs'])
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
            self._temp_data['local_weights'] = self.keras_layer._trainable_weights

            for var in self._temp_data['local_weights']:  # Keras does not add the vars to the collection
                if var.trainable and var not in tf.get_collection(TF_GRAPHKEYS_VARIABLES):
                    tf.add_to_collection(name=TF_GRAPHKEYS_VARIABLES, value=var)
