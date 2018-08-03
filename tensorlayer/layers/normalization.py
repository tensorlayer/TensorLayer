#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect

import tensorflow as tf
from tensorflow.python.training import moving_averages

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import force_return_self

__all__ = [
    'LocalResponseNormLayer',
    'BatchNormLayer',
    'InstanceNormLayer',
    'LayerNormLayer',
    'SwitchNormLayer',
]


class LocalResponseNormLayer(Layer):
    """The :class:`LocalResponseNormLayer` layer is for Local Response Normalization.
    See ``tf.nn.local_response_normalization`` or ``tf.nn.lrn`` for new TF version.
    The 4-D input tensor is a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.
    Within a given vector, each component is divided by the weighted square-sum of inputs within depth_radius.

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer with a 4D output shape.
    depth_radius : int
        Depth radius. 0-D. Half-width of the 1-D normalization window.
    bias : float
        An offset which is usually positive and shall avoid dividing by 0.
    alpha : float
        A scale factor which is usually positive.
    beta : float
        An exponent.
    name : str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            depth_radius=None,
            bias=None,
            alpha=None,
            beta=None,
            name='lrn_layer',
    ):

        self.prev_layer = prev_layer
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.name = name

        super(LocalResponseNormLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("depth_radius: %s" % self.depth_radius)
        except AttributeError:
            pass

        try:
            additional_str.append("bias: %s" % self.bias)
        except AttributeError:
            pass

        try:
            additional_str.append("alpha: %s" % self.alpha)
        except AttributeError:
            pass

        try:
            additional_str.append("beta: %s" % self.beta)
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(LocalResponseNormLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name):
            self.outputs = tf.nn.local_response_normalization(
                self.inputs, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta
            )

        self._add_layers(self.outputs)


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    decay : float
        A decay factor for `ExponentialMovingAverage`.
        Suggest to use a large value for large dataset.
    epsilon : float
        Eplison.
    act : activation function
        The activation function of this layer.
    is_train : boolean
        Is being used for training or inference.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    name : str
        A unique layer name.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            decay=0.9,
            epsilon=1e-5,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            moving_mean_init=tf.zeros_initializer,
            moving_var_init=tf.constant_initializer(1.),
            act=None,
            is_train=False,
            name='batchnorm_layer',
    ):

        self.prev_layer = prev_layer
        self.decay = decay
        self.epsilon = epsilon
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.act = act
        self.is_train = is_train
        self.name = name

        for initializer in ['beta_init', 'gamma_init', 'moving_mean_init', 'moving_var_init']:
            _init = getattr(self, initializer)
            if inspect.isclass(_init):
                setattr(self, initializer, _init())

        super(BatchNormLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("decay: %s" % self.decay)
        except AttributeError:
            pass

        try:
            additional_str.append("epsilon: %s" % self.epsilon)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(BatchNormLayer, self).__call__(prev_layer)

        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        with tf.variable_scope(self.name):
            axis = list(range(len(x_shape) - 1))

            # 1. beta, gamma

            if self.beta_init:

                beta = self._get_tf_variable(
                    'beta', shape=params_shape, initializer=self.beta_init, dtype=self.inputs.dtype, trainable=is_train
                )

            else:
                beta = None

            if self.gamma_init:
                gamma = self._get_tf_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=self.gamma_init,
                    dtype=self.inputs.dtype,
                    trainable=is_train,
                )
            else:
                gamma = None

            # 2.

            moving_mean = self._get_tf_variable(
                'moving_mean', params_shape, initializer=self.moving_mean_init, dtype=self.inputs.dtype, trainable=False
            )

            moving_variance = self._get_tf_variable(
                'moving_variance',
                params_shape,
                initializer=self.moving_var_init,
                dtype=self.inputs.dtype,
                trainable=False,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            self.outputs = self._apply_activation(
                tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, self.epsilon)
            )

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


class InstanceNormLayer(Layer):
    """The :class:`InstanceNormLayer` class is a for instance normalization.

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer.
    act : activation function.
        The activation function of this layer.
    epsilon : float
        Eplison.
    name : str
        A unique layer name

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            epsilon=1e-5,
            act=None,
            name='instance_norm',
    ):

        self.prev_layer = prev_layer
        self.epsilon = epsilon
        self.act = act
        self.name = name

        super(InstanceNormLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("epsilon: %s" % self.epsilon)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        if len(prev_layer.outputs.shape) not in [3, 4]:
            raise RuntimeError("`%s` only accepts input Tensor of dimension 3 or 4." % self.__class__.__name__)

        super(InstanceNormLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name):
            mean, var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)

            scale = self._get_tf_variable(
                'scale', [self.inputs.get_shape()[-1]], dtype=self.inputs.dtype,
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02)
            )

            offset = self._get_tf_variable(
                'offset', [self.inputs.get_shape()[-1]], dtype=self.inputs.dtype,
                initializer=tf.constant_initializer(0.0)
            )

            self.outputs = tf.div(self.inputs - mean, tf.sqrt(var + self.epsilon))
            self.outputs = tf.multiply(scale, tf.div(self.inputs - mean, tf.sqrt(var + self.epsilon)))
            self.outputs = tf.add(self.outputs, offset)

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


class LayerNormLayer(Layer):
    """
    The :class:`LayerNormLayer` class is for layer normalization, see `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    act : activation function
        The activation function of this layer.
    others : _
        `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self, prev_layer=None, center=True, scale=True, variables_collections=None, outputs_collections=None,
            begin_norm_axis=1, begin_params_axis=-1, act=None, name='layernorm'
    ):

        self.prev_layer = prev_layer
        self.center = center
        self.scale = scale
        self.variables_collections = variables_collections
        self.outputs_collections = outputs_collections
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.act = act
        self.name = name

        super(LayerNormLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("center: %s" % self.center)
        except AttributeError:
            pass

        try:
            additional_str.append("scale: %s" % self.scale)
        except AttributeError:
            pass

        try:
            additional_str.append("variables_collections: %s" % self.variables_collections)
        except AttributeError:
            pass

        try:
            additional_str.append("outputs_collections: %s" % self.outputs_collections)
        except AttributeError:
            pass

        try:
            additional_str.append("begin_norm_axis: %s" % self.begin_norm_axis)
        except AttributeError:
            pass

        try:
            additional_str.append("begin_params_axis: %s" % self.begin_params_axis)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        super(LayerNormLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name) as vs:
            self.outputs = tf.contrib.layers.layer_norm(
                self.inputs,
                center=self.center,
                scale=self.scale,
                activation_fn=None,
                variables_collections=self.variables_collections,
                outputs_collections=self.outputs_collections,
                trainable=is_train,
                begin_norm_axis=self.begin_norm_axis,
                begin_params_axis=self.begin_params_axis,
                scope='var',
            )

            self.outputs = self._apply_activation(self.outputs)

            self._local_weights = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)


class SwitchNormLayer(Layer):
    """
    The :class:`SwitchNormLayer` is a switchable normalization.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    act : activation function
        The activation function of this layer.
    epsilon : float
        Eplison.
    beta_init : initializer or None
        The initializer for initializing beta, if None, skip beta.
        Usually you should not skip beta unless you know what happened.
    gamma_init : initializer or None
        The initializer for initializing gamma, if None, skip gamma.
        When the batch normalization layer is use instead of 'biases', or the next layer is linear, this can be
        disabled since the scaling can be done by the next layer. see `Inception-ResNet-v2 <https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py>`__
    name : str
        A unique layer name.

    References
    ----------
    - `Differentiable Learning-to-Normalize via Switchable Normalization <https://arxiv.org/abs/1806.10779>`__
    - `Zhihu (CN) <https://zhuanlan.zhihu.com/p/39296570?utm_source=wechat_session&utm_medium=social&utm_oi=984862267107651584>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer=None,
            epsilon=1e-5,
            beta_init=tf.constant_initializer(0.0),
            gamma_init=tf.constant_initializer(1.0),
            act=None,
            name='switchnorm_layer',
    ):

        self.prev_layer = prev_layer
        self.epsilon = epsilon
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.act = act
        self.name = name

        super(SwitchNormLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("epsilon: %s" % self.epsilon)
        except AttributeError:
            pass

        try:
            additional_str.append("act: %s" % self.act.__name__ if self.act is not None else 'No Activation')
        except AttributeError:
            pass

        return self._str(additional_str)

    @force_return_self
    def __call__(self, prev_layer, is_train=True):

        if len(prev_layer.outputs.shape) not in [3, 4]:
            raise RuntimeError("`%s` only accepts input Tensor of dimension 3 or 4." % self.__class__.__name__)

        super(SwitchNormLayer, self).__call__(prev_layer)

        with tf.variable_scope(self.name):

            ch = self.inputs.shape[-1]

            batch_mean, batch_var = tf.nn.moments(self.inputs, [0, 1, 2], keep_dims=True)
            ins_mean, ins_var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)
            layer_mean, layer_var = tf.nn.moments(self.inputs, [1, 2, 3], keep_dims=True)

            gamma = self._get_tf_variable("gamma", [ch], initializer=self.gamma_init)
            beta = self._get_tf_variable("beta", [ch], initializer=self.beta_init)

            mean_weight_var = self._get_tf_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0))
            var_weight_var = self._get_tf_variable("var_weight", [3], initializer=tf.constant_initializer(1.0))

            mean_weight = tf.nn.softmax(mean_weight_var)
            var_weight = tf.nn.softmax(var_weight_var)

            mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
            var = var_weight[0] * batch_var + var_weight[1] * ins_var + var_weight[2] * layer_var

            self.outputs = (self.inputs - mean) / (tf.sqrt(var + self.epsilon))

            self.outputs = tf.add(tf.multiply(self.inputs, gamma), beta)
            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        self._add_params(self._local_weights)
