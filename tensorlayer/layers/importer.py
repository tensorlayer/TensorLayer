#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.util.deprecation import deprecated

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'LambdaLayer',
    'SlimNetsLayer',
    'KerasLayer',
    'EstimatorLayer',
]


class LambdaLayer(Layer):
    """A layer that takes a user-defined function using TensorFlow Lambda, for multiple inputs see :class:`ElementwiseLambdaLayer`.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    fn : function
        The function that applies to the outputs of previous layer.
    fn_args : dictionary or None
        The arguments for the function (option).
    name : str
        A unique layer name.

    Examples
    ---------
    Non-parametric case

    >>> x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = LambdaLayer(net, lambda x: 2*x, name='lambda')

    Parametric case, merge other wrappers into TensorLayer

    >>> from keras.layers import *
    >>> from tensorlayer.layers import *
    >>> def keras_block(x):
    >>>     x = Dropout(0.8)(x)
    >>>     x = Dense(800, activation='relu')(x)
    >>>     x = Dropout(0.5)(x)
    >>>     x = Dense(800, activation='relu')(x)
    >>>     x = Dropout(0.5)(x)
    >>>     logits = Dense(10, activation='linear')(x)
    >>>     return logits
    >>> net = InputLayer(x, name='input')
    >>> net = LambdaLayer(net, fn=keras_block, name='keras')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            fn,
            fn_args=None,
            name='lambda_layer',
    ):

        super(LambdaLayer, self).__init__(prev_layer=prev_layer, fn_args=fn_args, name=name)

        logging.info("LambdaLayer  %s" % self.name)

        if fn is None:
            raise AssertionError("The `fn` argument cannot be None")

        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **self.fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


class SlimNetsLayer(Layer):
    """A layer that merges TF-Slim models into TensorLayer.

    Models can be found in `slim-model <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`__,
    see Inception V3 example on `Github <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`__.

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
            prev_layer,
            slim_layer,
            slim_args=None,
            name='tfslim_layer',
    ):

        if slim_layer is None:
            raise ValueError("slim layer is None")

        super(SlimNetsLayer, self).__init__(prev_layer=prev_layer, slim_args=slim_args, name=name)

        logging.info("SlimNetsLayer %s: %s" % (self.name, slim_layer.__name__))

        # with tf.variable_scope(name) as vs:
        #     net, end_points = slim_layer(self.inputs, **slim_args)
        #     slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        with tf.variable_scope(name):
            self.outputs, end_points = slim_layer(self.inputs, **self.slim_args)

        slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=self.name)

        if slim_variables == []:
            logging.error(
                "No variables found under %s : the name of SlimNetsLayer should be matched with the begining of the ckpt file, see tutorial_inceptionV3_tfslim.py for more details"
                % self.name
            )

        slim_layers = []

        for v in end_points.values():
            # tf.contrib.layers.summaries.summarize_activation(v)
            slim_layers.append(v)

        self._add_layers(slim_layers)
        self._add_params(slim_variables)


@deprecated("2018-06-30", "This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.")
class KerasLayer(Layer):
    """A layer to import Keras layers into TensorLayer.

    Example can be found here `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`__.

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
            prev_layer,
            keras_layer,
            keras_args=None,
            name='keras_layer',
    ):

        super(KerasLayer, self).__init__(prev_layer=prev_layer, keras_args=keras_args, name=name)

        logging.info("KerasLayer %s: %s" % (self.name, keras_layer))

        logging.warning("This API will be removed, please use LambdaLayer instead.")

        with tf.variable_scope(name) as vs:
            self.outputs = keras_layer(self.inputs, **self.keras_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


@deprecated("2018-06-30", "This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.")
class EstimatorLayer(Layer):
    """A layer that accepts a user-defined model.

    It is similar with :class:`KerasLayer`, see `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`__.

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
            prev_layer,
            model_fn,
            layer_args=None,
            name='estimator_layer',
    ):
        super(EstimatorLayer, self).__init__(prev_layer=prev_layer, layer_args=layer_args, name=name)

        logging.info("EstimatorLayer %s: %s" % (self.name, model_fn))

        if model_fn is None:
            raise ValueError('model fn is None')

        logging.warning("This API will be removed, please use LambdaLayer instead.")

        with tf.variable_scope(name) as vs:
            self.outputs = model_fn(self.inputs, **self.layer_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)
