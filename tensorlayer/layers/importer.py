# -*- coding: utf-8 -*-

from tensorflow.python.util.deprecation import deprecated

from .core import *


class LambdaLayer(Layer):
    """A layer that takes a user-defined function using TensorFlow Lambda.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
            fn,
            fn_args=None,
            name='lambda_layer',
    ):
        if fn_args is None:
            fn_args = {}
        Layer.__init__(self, name=name)
        assert layer is not None
        assert fn is not None
        self.inputs = layer.outputs
        logging.info("LambdaLayer  %s" % self.name)
        with tf.variable_scope(name) as vs:
            self.outputs = fn(self.inputs, **fn_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


class SlimNetsLayer(Layer):
    """A layer that merges TF-Slim models into TensorLayer.

    Models can be found in `slim-model <https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models>`__,
    see Inception V3 example on `Github <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_inceptionV3_tfslim.py>`__.

    Parameters
    ----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            layer,
            slim_layer,
            slim_args=None,
            name='tfslim_layer',
    ):
        if slim_layer is None:
            raise ValueError("slim layer is None")
        if slim_args is None:
            slim_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("SlimNetsLayer %s: %s" % (self.name, slim_layer.__name__))

        # with tf.variable_scope(name) as vs:
        #     net, end_points = slim_layer(self.inputs, **slim_args)
        #     slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        net, end_points = slim_layer(self.inputs, **slim_args)

        slim_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=name)
        if slim_variables == []:
            logging.info(
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


@deprecated("2018-06-30", "This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.")
class KerasLayer(Layer):
    """A layer to import Keras layers into TensorLayer.

    Example can be found here `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`__.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    keras_layer : function
        A tensor in tensor out function for building model.
    keras_args : dictionary
        The arguments for the `keras_layer`.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
            keras_layer,
            keras_args=None,
            name='keras_layer',
    ):
        if layer is None:
            raise ValueError("layer is None")
        if keras_args is None:
            keras_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("KerasLayer %s: %s" % (self.name, keras_layer))
        logging.info("This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = keras_layer(self.inputs, **keras_args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


@deprecated("2018-06-30", "This layer will be deprecated soon as :class:`LambdaLayer` can do the same thing.")
class EstimatorLayer(Layer):
    """A layer that accepts a user-defined model.

    It is similar with :class:`KerasLayer`, see `tutorial_keras.py <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py>`__.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    model_fn : function
        A tensor in tensor out function for building model.
    args : dictionary
        The arguments for the `model_fn`.
    name : str
        A unique layer name.

    """

    def __init__(
            self,
            layer,
            model_fn,
            args=None,
            name='estimator_layer',
    ):
        if model_fn is None:
            raise ValueError('model fn is None')
        if args is None:
            args = {}
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        logging.info("EstimatorLayer %s: %s" % (self.name, model_fn))
        logging.info("This API will be removed, please use LambdaLayer instead.")
        with tf.variable_scope(name) as vs:
            self.outputs = model_fn(self.inputs, **args)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
