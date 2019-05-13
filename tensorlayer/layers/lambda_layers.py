#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.decorators import deprecated_alias
from tensorlayer.layers.core import Layer
from tensorlayer.files import utils

# from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

__all__ = [
    'Lambda',
    'ElementwiseLambda',
]


class Lambda(Layer):
    """A layer that takes a user-defined function using Lambda.
    If the function has trainable weights, the weights should be provided.
    Remember to make sure the weights provided when the layer is constructed are SAME as
    the weights used when the layer is forwarded.
    For multiple inputs see :class:`ElementwiseLambda`.

    Parameters
    ----------
    fn : function
        The function that applies to the inputs (e.g. tensor from the previous layer).
    fn_weights : list
        The trainable weights for the function if any. Optional.
    fn_args : dict
        The arguments for the function if any. Optional.
    name : str or None
        A unique layer name.

    Examples
    ---------
    Non-parametric and non-args case
    This case is supported in the Model.save() / Model.load() to save / load the whole model architecture and weights(optional).

    >>> x = tl.layers.Input([8, 3], name='input')
    >>> y = tl.layers.Lambda(lambda x: 2*x, name='lambda')(x)


    Non-parametric and with args case
    This case is supported in the Model.save() / Model.load() to save / load the whole model architecture and weights(optional).

    >>> def customize_func(x, foo=42): # x is the inputs, foo is an argument
    >>>     return foo * x
    >>> x = tl.layers.Input([8, 3], name='input')
    >>> lambdalayer = tl.layers.Lambda(customize_func, fn_args={'foo': 2}, name='lambda')(x)


    Any function with outside variables
    This case has not been supported in Model.save() / Model.load() yet.
    Please avoid using Model.save() / Model.load() to save / load models that contain such Lambda layer. Instead, you may use Model.save_weights() / Model.load_weights() to save / load model weights.
    Note: In this case, fn_weights should be a list, and then the trainable weights in this Lambda layer can be added into the weights of the whole model.

    >>> vara = [tf.Variable(1.0)]
    >>> def func(x):
    >>>     return x + vara
    >>> x = tl.layers.Input([8, 3], name='input')
    >>> y = tl.layers.Lambda(func, fn_weights=a, name='lambda')(x)


    Parametric case, merge other wrappers into TensorLayer
    This case is supported in the Model.save() / Model.load() to save / load the whole model architecture and weights(optional).

    >>> layers = [
    >>>     tf.keras.layers.Dense(10, activation=tf.nn.relu),
    >>>     tf.keras.layers.Dense(5, activation=tf.nn.sigmoid),
    >>>     tf.keras.layers.Dense(1, activation=tf.identity)
    >>> ]
    >>> perceptron = tf.keras.Sequential(layers)
    >>> # in order to compile keras model and get trainable_variables of the keras model
    >>> _ = perceptron(np.random.random([100, 5]).astype(np.float32))

    >>> class CustomizeModel(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(CustomizeModel, self).__init__()
    >>>         self.dense = tl.layers.Dense(in_channels=1, n_units=5)
    >>>         self.lambdalayer = tl.layers.Lambda(perceptron, perceptron.trainable_variables)

    >>>     def forward(self, x):
    >>>         z = self.dense(x)
    >>>         z = self.lambdalayer(z)
    >>>         return z

    >>> optimizer = tf.optimizers.Adam(learning_rate=0.1)
    >>> model = CustomizeModel()
    >>> model.train()

    >>> for epoch in range(50):
    >>>     with tf.GradientTape() as tape:
    >>>         pred_y = model(data_x)
    >>>         loss = tl.cost.mean_squared_error(pred_y, data_y)

    >>>     gradients = tape.gradient(loss, model.trainable_weights)
    >>>     optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    """

    def __init__(
            self,
            fn,
            fn_weights=None,
            fn_args=None,
            name=None,
    ):

        super(Lambda, self).__init__(name=name)
        self.fn = fn
        self._trainable_weights = fn_weights if fn_weights is not None else []
        self.fn_args = fn_args if fn_args is not None else {}

        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        logging.info("Lambda  %s: func: %s, len_weights: %s" % (self.name, fn_name, len(self._trainable_weights)))

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'fn={fn_name},'
        s += 'len_weights={len_weights},'
        s += 'name=\'{name}\''
        s += ')'
        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        return s.format(
            classname=self.__class__.__name__, fn_name=fn_name, len_weights=len(self._trainable_weights),
            **self.__dict__
        )

    def build(self, inputs_shape=None):
        pass

    def forward(self, inputs, **kwargs):

        if len(kwargs) == 0:
            outputs = self.fn(inputs, **self.fn_args)
        else:
            outputs = self.fn(inputs, **kwargs)

        return outputs

    def get_args(self):
        init_args = {}
        if isinstance(self.fn, tf.keras.layers.Layer) or isinstance(self.fn, tf.keras.Model):
            init_args.update({"layer_type": "keraslayer"})
            init_args["fn"] = utils.save_keras_model(self.fn)
            init_args["fn_weights"] = None
            if len(self._nodes) == 0:
                init_args["keras_input_shape"] = []
            else:
                init_args["keras_input_shape"] = self._nodes[0].in_tensors[0].get_shape().as_list()
        else:
            init_args = {"layer_type": "normal"}
        return init_args


class ElementwiseLambda(Layer):
    """A layer that use a custom function to combine multiple :class:`Layer` inputs.
    If the function has trainable weights, the weights should be provided.
    Remember to make sure the weights provided when the layer is constructed are SAME as
    the weights used when the layer is forwarded.

    Parameters
    ----------
    fn : function
        The function that applies to the inputs (e.g. tensor from the previous layer).
    fn_weights : list
        The trainable weights for the function if any. Optional.
    fn_args : dict
        The arguments for the function if any. Optional.
    name : str or None
        A unique layer name.

    Examples
    --------

    Non-parametric and with args case
    This case is supported in the Model.save() / Model.load() to save / load the whole model architecture and weights(optional).

    z = mean + noise * tf.exp(std * 0.5) + foo
    >>> def func(noise, mean, std, foo=42):
    >>>     return mean + noise * tf.exp(std * 0.5) + foo

    >>> noise = tl.layers.Input([100, 1])
    >>> mean = tl.layers.Input([100, 1])
    >>> std = tl.layers.Input([100, 1])
    >>> out = tl.layers.ElementwiseLambda(fn=func, fn_args={'foo': 84}, name='elementwiselambda')([noise, mean, std])


    Non-parametric and non-args case
    This case is supported in the Model.save() / Model.load() to save / load the whole model architecture and weights(optional).

    z = mean + noise * tf.exp(std * 0.5)
    >>> noise = tl.layers.Input([100, 1])
    >>> mean = tl.layers.Input([100, 1])
    >>> std = tl.layers.Input([100, 1])
    >>> out = tl.layers.ElementwiseLambda(fn=lambda x, y, z: x + y * tf.exp(z * 0.5), name='elementwiselambda')([noise, mean, std])


    Any function with outside variables
    This case has not been supported in Model.save() / Model.load() yet.
    Please avoid using Model.save() / Model.load() to save / load models that contain such ElementwiseLambda layer. Instead, you may use Model.save_weights() / Model.load_weights() to save / load model weights.
    Note: In this case, fn_weights should be a list, and then the trainable weights in this ElementwiseLambda layer can be added into the weights of the whole model.

    z = mean + noise * tf.exp(std * 0.5) + vara
    >>> vara = [tf.Variable(1.0)]
    >>> def func(noise, mean, std):
    >>>     return mean + noise * tf.exp(std * 0.5) + vara
    >>> noise = tl.layers.Input([100, 1])
    >>> mean = tl.layers.Input([100, 1])
    >>> std = tl.layers.Input([100, 1])
    >>> out = tl.layers.ElementwiseLambda(fn=func, fn_weights=vara, name='elementwiselambda')([noise, mean, std])

    """

    def __init__(
            self,
            fn,
            fn_weights=None,
            fn_args=None,
            name=None,  #'elementwiselambda',
    ):

        super(ElementwiseLambda, self).__init__(name=name)
        self.fn = fn
        self._trainable_weights = fn_weights if fn_weights is not None else []
        self.fn_args = fn_args if fn_args is not None else {}

        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        logging.info(
            "ElementwiseLambda  %s: func: %s, len_weights: %s" % (self.name, fn_name, len(self._trainable_weights))
        )

        self.build()
        self._built = True

    def __repr__(self):
        s = '{classname}('
        s += 'fn={fn_name},'
        s += 'len_weights={len_weights},'
        s += 'name=\'{name}\''
        s += ')'
        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        return s.format(
            classname=self.__class__.__name__, fn_name=fn_name, len_weights=len(self._trainable_weights),
            **self.__dict__
        )

    def build(self, inputs_shape=None):
        # do nothing
        # the weights of the function are provided when the Lambda layer is constructed
        pass

    # @tf.function
    def forward(self, inputs, **kwargs):

        if not isinstance(inputs, list):
            raise TypeError(
                "The inputs should be a list of values which corresponds with the customised lambda function."
            )

        if len(kwargs) == 0:
            outputs = self.fn(*inputs, **self.fn_args)
        else:
            outputs = self.fn(*inputs, **kwargs)

        return outputs
