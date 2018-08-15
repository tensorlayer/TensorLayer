#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

__all__ = [
    'SlimNetsLayer',
    'KerasLayer',
    'EstimatorLayer',
]


class SlimNetsLayer(Layer):
    """A layer that merges TF-Slim models into TensorLayer.

    Models can be found in `slim-model <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`__,
    see Inception V3 example on `Github <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            slim_layer=None,
            slim_args=None,
            act=None,
            name='tfslim_layer',
    ):

        if slim_layer is None:
            raise ValueError("slim layer is None")

        self.prev_layer = prev_layer
        self.slim_layer = slim_layer
        self.act = act
        self.name = name

        super(SlimNetsLayer, self).__init__(slim_args=slim_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer kind: %s" % self.slim_layer.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(SlimNetsLayer, self).__call__(prev_layer)

        slim_layers = []

        with tf.variable_scope(self.name) as vs:

            _out = self.slim_layer(self.inputs, **self.slim_args)

            if isinstance(_out, tf.Tensor):
                self.outputs = _out
                slim_layers.append(_out)

            else:
                self.outputs, end_points = _out

                for v in end_points.values():
                    slim_layers.append(v)

            slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(slim_layers)
        self._add_params(slim_variables)


@deprecated(
    date="2018-06-30", instructions="This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing"
)
class KerasLayer(Layer):
    """A layer to import Keras layers into TensorLayer.

    Example can be found here `tutorial_keras.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_keras.py>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    keras_layer : function
        A tensor in tensor out function for building model.
    keras_args : dictionary
        The arguments for the `keras_layer`.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            keras_layer=None,
            keras_args=None,
            act=None,
            name='keras_layer',
    ):

        if keras_layer is None:
            raise ValueError("keras_layer is None")

        self.prev_layer = prev_layer
        self.keras_layer = keras_layer(**keras_args)
        self.act = act
        self.name = name

        super(KerasLayer, self).__init__(keras_args=keras_args)

        logging.warning("This API will be removed, please use `LambdaLayer` instead.")

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer kind: %s" % self.keras_layer.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(KerasLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name) as vs:
            self.outputs = self.keras_layer(self.inputs)
            keras_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        self._add_params(keras_variables)


@deprecated(
    date="2018-06-30", instructions="This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing"
)
class EstimatorLayer(Layer):
    """A layer that accepts a user-defined model.

    It is similar with :class:`KerasLayer`, see `tutorial_keras.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_keras.py>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer
    model_fn : function
        A tensor in tensor out function for building model.
    layer_args : dictionary
        The arguments for the `model_fn`.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', args='layer_args', end_support_version=1.9
    )  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            model_fn=None,
            layer_args=None,
            act=None,
            name='estimator_layer',
    ):

        if model_fn is None:
            raise ValueError("model_fn is None")

        self.prev_layer = prev_layer
        self.model_fn = model_fn
        self.act = act
        self.name = name

        super(EstimatorLayer, self).__init__(layer_args=layer_args)

        logging.warning("This API will be removed, please use `LambdaLayer` instead.")

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("layer kind: %s" % self.model_fn.__name__)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(EstimatorLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name) as vs:
            self.outputs = self.model_fn(self.inputs, **self.layer_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)
