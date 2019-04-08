#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer import logging
from tensorlayer.layers.core import Layer

__all__ = [
    'Concat',
    'Elementwise',
]


class Concat(Layer):
    """A layer that concats multiple tensors according to given axis.

    Parameters
    ----------
    concat_dim : int
        The dimension to concatenate.
    name : None or str
        A unique layer name.

    Examples
    ----------
    >>> class CustomModel(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(CustomModel, self).__init__(name="custom")
    >>>         self.dense1 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu1_1')
    >>>         self.dense2 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu2_1')
    >>>         self.concat = tl.layers.Concat(concat_dim=1, name='concat_layer')

    >>>     def forward(self, inputs):
    >>>         d1 = self.dense1(inputs)
    >>>         d2 = self.dense2(inputs)
    >>>         outputs = self.concat([d1, d2])
    >>>         return outputs

    """

    def __init__(
            self,
            concat_dim=-1,
            name=None,  #'concat',
    ):

        super(Concat, self).__init__(name)
        self.concat_dim = concat_dim

        self.build(None)
        self._built = True

        logging.info("Concat %s: concat_dim: %d" % (self.name, concat_dim))

    def __repr__(self):
        s = ('{classname}(concat_dim={concat_dim})')
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        pass

    # @tf.function
    def forward(self, inputs):
        """

        prev_layer : list of :class:`Layer`
            List of layers to concatenate.
        """
        outputs = tf.concat(inputs, self.concat_dim, name=self.name)

        return outputs


class Elementwise(Layer):
    """A layer that combines multiple :class:`Layer` that have the same output shapes
    according to an element-wise operation.
    If the element-wise operation is complicated, please consider to use :class:`ElementwiseLambda`.

    Parameters
    ----------
    combine_fn : a TensorFlow element-wise combine function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`__ .
        If the combine function is more complicated, please consider to use :class:`ElementwiseLambda`.
    act : activation function
        The activation function of this layer.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> class CustomModel(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(CustomModel, self).__init__(name="custom")
    >>>         self.dense1 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu1_1')
    >>>         self.dense2 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu2_1')
    >>>         self.element = tl.layers.Elementwise(combine_fn=tf.minimum, name='minimum', act=tf.identity)

    >>>     def forward(self, inputs):
    >>>         d1 = self.dense1(inputs)
    >>>         d2 = self.dense2(inputs)
    >>>         outputs = self.element([d1, d2])
    >>>         return outputs
    """

    def __init__(
            self,
            combine_fn=tf.minimum,
            act=None,
            name=None,  #'elementwise',
    ):

        super(Elementwise, self).__init__(name)
        self.combine_fn = combine_fn
        self.act = act

        self.build(None)
        self._built = True

        logging.info(
            "Elementwise %s: fn: %s act: %s" %
            (self.name, combine_fn.__name__, ('No Activation' if self.act is None else self.act.__name__))
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(combine_fn={combine_fn}, ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = inputs[0]
        for input in inputs[1:]:
            outputs = self.combine_fn(outputs, input, name=self.name)
        if self.act:
            outputs = self.act(outputs)
        return outputs
