#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'Lambda',
    'ElementwiseLambda',
]


class Lambda(Layer):
    """A layer that takes a user-defined function using Lambda.
    If the function has trainable weights, the weights should be provided.
    Remember to make sure the weights provided when the layer is constructed are SAME as
    the weights used when the layer is forwarded.
    For multiple inputs see :class:`ElementwiseLambdaLayer`.

    Parameters
    ----------
    fn : function
        The function that applies to the inputs (e.g. tensor from the previous layer).
    fn_weights: a list of trainable weights (e.g. tf.Variable)
        Optional. If the function has trainable weights, the weights should be explicitly provided.
        Remember to make sure the weights provided when the layer is constructed are SAME as
        the weights used when the layer is forwarded.
    fn_args: a dict
        Optional, the arguments for the function if any.
        In a dynamic model, fn_args can be given via **kwargs when the layer is called.
        Note that the arguments should not be inputs.
        For multiple inputs, see :class:`ElementwiseLambdaLayer`.
    name : str or None
        A unique layer name.

    Examples
    ---------
    Non-parametric case

    >>> x = tl.layers.Input([8, 3], name='input')
    >>> y = tl.layers.Lambda(lambda x: 2*x, name='lambda')(x)

    >>> def customize_func(x, foo=42): # x is the inputs, foo is an argument
    >>>     return foo * x
    >>> x = tl.layers.Input([8, 3], name='input')
    >>> lambdalayer = tl.layers.Lambda(customize_func, fn_args={'foo': 2}, name='lambda')(x)

    Parametric case, merge other wrappers into TensorLayer

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

    >>>     gradients = tape.gradient(loss, model.weights)
    >>>     optimizer.apply_gradients(zip(gradients, model.weights))

    """

    def __init__(
            self,
            fn,
            fn_weights=None,
            fn_args=None,
            name=None,  #'lambda',
    ):

        super(Lambda, self).__init__(name=name)
        self.fn = fn
        self._weights = fn_weights if fn_weights is not None else []
        self.fn_args = fn_args if fn_args is not None else {}

        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        logging.info("Lambda  %s: func: %s, len_weights: %s" % (self.name, fn_name, len(self._weights)))

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
            classname=self.__class__.__name__, fn_name=fn_name, len_weights=len(self._weights), **self.__dict__
        )

    def build(self, inputs_shape=None):
        # do nothing
        # the weights of the function are provided when the Lambda layer is constructed
        pass

    @tf.function
    def forward(self, inputs, **kwargs):

        if len(kwargs) == 0:
            outputs = self.fn(inputs, **self.fn_args)
        else:
            outputs = self.fn(inputs, **kwargs)

        return outputs


class ElementwiseLambda(Layer):
    """A layer that use a custom function to combine multiple :class:`Layer` inputs.
    If the function has trainable weights, the weights should be provided.
    Remember to make sure the weights provided when the layer is constructed are SAME as
    the weights used when the layer is forwarded.

    Parameters
    ----------
    fn : function
        The function that applies to the inputs (e.g. tensor from the previous layer).
    fn_weights: a list of trainable weights (e.g. tf.Variable)
        Optional. If the function has trainable weights, the weights should be explicitly provided.
        Remember to make sure the weights provided when the layer is constructed are SAME as
        the weights used when the layer is forwarded.
    fn_args: a dict
        Optional, the arguments for the function if any.
        In a dynamic model, fn_args can be given via **kwargs when the layer is called.
        Note that the arguments should not be inputs.
    name : str or None
        A unique layer name.

    Examples
    --------
    z = mean + noise * tf.exp(std * 0.5) + foo

    >>> def func(noise, mean, std, foo=42):
    >>>     return mean + noise * tf.exp(std * 0.5) + foo

    >>> noise = tl.layers.Input([100, 1])
    >>> mean = tl.layers.Input([100, 1])
    >>> std = tl.layers.Input([100, 1])
    >>> out = tl.layers.ElementwiseLambda(fn=func, fn_args={'foo': 84}, name='elementwiselambda')([noise, mean, std])
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
        self._weights = fn_weights if fn_weights is not None else []
        self.fn_args = fn_args if fn_args is not None else {}

        try:
            fn_name = repr(self.fn)
        except:
            fn_name = 'name not available'
        logging.info("ElementwiseLambda  %s: func: %s, len_weights: %s" % (self.name, fn_name, len(self._weights)))

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
            classname=self.__class__.__name__, fn_name=fn_name, len_weights=len(self._weights), **self.__dict__
        )

    def build(self, inputs_shape=None):
        # do nothing
        # the weights of the function are provided when the Lambda layer is constructed
        pass

    @tf.function
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
