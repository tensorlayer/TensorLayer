#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES
from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'LocalResponseNormLayer',
    'BatchNormLayer',
    'InstanceNormLayer',
    'LayerNormLayer',
    'GroupNormLayer',
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
            prev_layer,
            depth_radius=None,
            bias=None,
            alpha=None,
            beta=None,
            name='lrn_layer',
    ):
        super(LocalResponseNormLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "LocalResponseNormLayer %s: depth_radius: %s, bias: %s, alpha: %s, beta: %s" %
            (self.name, str(depth_radius), str(bias), str(alpha), str(beta))
        )

        with tf.variable_scope(name):
            self.outputs = tf.nn.lrn(self.inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

        self._add_layers(self.outputs)


def _to_channel_first_bias(b):
    """Reshape [c] to [c, 1, 1]."""
    channel_size = int(b.shape[0])
    new_shape = (channel_size, 1, 1)
    # new_shape = [-1, 1, 1]  # doesn't work with tensorRT
    return tf.reshape(b, new_shape)


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    if data_format == 'NHWC':
        return x * b
    elif data_format == 'NCHW':
        return x * _to_channel_first_bias(b)
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    if data_format == 'NHWC':
        return tf.add(x, b)
    elif data_format == 'NCHW':
        return tf.add(x, _to_channel_first_bias(b))
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    with ops.name_scope(name, 'batchnorm', [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale

        a = math_ops.cast(inv, x.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype)

        # Return a * x + b with customized data_format.
        # Currently TF doesn't have bias_scale, and tensorRT has bug in converting tf.nn.bias_add
        # So we reimplemted them to allow make the model work with tensorRT.
        # See https://github.com/tensorlayer/openpose-plus/issues/75 for more details.
        df = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
        return _bias_add(_bias_scale(x, a, df[data_format]), b, df[data_format])


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
            prev_layer,
            decay=0.9,
            epsilon=0.00001,
            act=None,
            is_train=False,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            moving_mean_init=tf.zeros_initializer(),
            data_format='channels_last',
            name='batchnorm_layer',
    ):
        super(BatchNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "BatchNormLayer %s: decay: %f epsilon: %f act: %s is_train: %s" %
            (self.name, decay, epsilon, self.act.__name__ if self.act is not None else 'No Activation', is_train)
        )
        if decay < 0 or 1 < decay:
            raise Exception("decay should be between 0 to 1")

        x_shape = self.inputs.get_shape()
        if data_format == 'channels_last':
            axis = len(x_shape) - 1
        elif data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))
        params_shape = x_shape[axis]

        with tf.variable_scope(name):
            axes = [i for i in range(len(x_shape)) if i != axis]

            # 1. beta, gamma
            variables = []

            if beta_init:

                if beta_init == tf.zeros_initializer:
                    beta_init = beta_init()

                beta = tf.get_variable(
                    'beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype, trainable=is_train
                )

                variables.append(beta)

            else:
                beta = None

            if gamma_init:
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=LayersConfig.tf_dtype,
                    trainable=is_train,
                )
                variables.append(gamma)
            else:
                gamma = None

            # 2.

            moving_mean = tf.get_variable(
                'moving_mean', params_shape, initializer=moving_mean_init, dtype=LayersConfig.tf_dtype, trainable=False
            )

            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axes)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            self.outputs = self._apply_activation(
                batch_normalization(self.inputs, mean, var, beta, gamma, epsilon, data_format)
            )

            variables.extend([moving_mean, moving_variance])

        self._add_layers(self.outputs)
        self._add_params(variables)


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
            prev_layer,
            act=None,
            epsilon=1e-5,
            name='instan_norm',
    ):
        super(InstanceNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "InstanceNormLayer %s: epsilon: %f act: %s" %
            (self.name, epsilon, self.act.__name__ if self.act is not None else 'No Activation')
        )

        with tf.variable_scope(name) as vs:
            mean, var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)

            scale = tf.get_variable(
                'scale', [self.inputs.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=LayersConfig.tf_dtype
            )

            offset = tf.get_variable(
                'offset', [self.inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0),
                dtype=LayersConfig.tf_dtype
            )

            self.outputs = scale * tf.div(self.inputs - mean, tf.sqrt(var + epsilon)) + offset
            self.outputs = self._apply_activation(self.outputs)

            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


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
            self, prev_layer, center=True, scale=True, act=None, reuse=None, variables_collections=None,
            outputs_collections=None, trainable=True, begin_norm_axis=1, begin_params_axis=-1, name='layernorm'
    ):

        super(LayerNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "LayerNormLayer %s: act: %s" % (self.name, self.act.__name__ if self.act is not None else 'No Activation')
        )

        with tf.variable_scope(name) as vs:
            self.outputs = tf.contrib.layers.layer_norm(
                self.inputs,
                center=center,
                scale=scale,
                activation_fn=self.act,
                reuse=reuse,
                variables_collections=variables_collections,
                outputs_collections=outputs_collections,
                trainable=trainable,
                begin_norm_axis=begin_norm_axis,
                begin_params_axis=begin_params_axis,
                scope='var',
            )

            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


class GroupNormLayer(Layer):
    """The :class:`GroupNormLayer` layer is for Group Normalization.
    See `tf.contrib.layers.group_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/group_norm>`__.

    Parameters
    -----------
    prev_layer : :class:`Layer`
        The previous layer.
    act : activation function
        The activation function of this layer.
    epsilon : float
        Eplison.
    name : str
        A unique layer name

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(self, prev_layer, groups=32, epsilon=1e-06, act=None, data_format='channels_last', name='groupnorm'):
        super(GroupNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "GroupNormLayer %s: act: %s" % (self.name, self.act.__name__ if self.act is not None else 'No Activation')
        )

        shape = self.inputs.get_shape().as_list()
        if len(shape) != 4:
            raise Exception("GroupNormLayer only supports 2D images.")

        if data_format == 'channels_last':
            channels = shape[-1]
            int_shape = tf.concat(
                [tf.shape(self.inputs)[0:3],
                 tf.convert_to_tensor([groups, channels // groups])], axis=0
            )
        elif data_format == 'channels_first':
            channels = shape[1]
            int_shape = tf.concat(
                [
                    tf.shape(self.inputs)[0:1],
                    tf.convert_to_tensor([groups, channels // groups]),
                    tf.shape(self.inputs)[2:4]
                ], axis=0
            )
        else:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")

        if groups > channels:
            raise ValueError('Invalid groups %d for %d channels.' % (groups, channels))
        if channels % groups != 0:
            raise ValueError('%d channels is not commensurate with %d groups.' % (channels, groups))

        with tf.variable_scope(name):
            x = tf.reshape(self.inputs, int_shape)
            if data_format == 'channels_last':
                mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
                gamma = tf.get_variable('gamma', channels, initializer=tf.ones_initializer())
                beta = tf.get_variable('beta', channels, initializer=tf.zeros_initializer())
            else:
                mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
                gamma = tf.get_variable('gamma', [1, channels, 1, 1], initializer=tf.ones_initializer())
                beta = tf.get_variable('beta', [1, channels, 1, 1], initializer=tf.zeros_initializer())

            x = (x - mean) / tf.sqrt(var + epsilon)

            self.outputs = tf.reshape(x, tf.shape(self.inputs)) * gamma + beta
            self.outputs = self._apply_activation(self.outputs)

        variables = get_collection_trainable(self.name)

        self._add_layers(self.outputs)
        self._add_params(variables)


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
            prev_layer,
            act=None,
            epsilon=1e-5,
            beta_init=tf.constant_initializer(0.0),
            gamma_init=tf.constant_initializer(1.0),
            moving_mean_init=tf.zeros_initializer(),
            name='switchnorm_layer',
    ):
        super(SwitchNormLayer, self).__init__(prev_layer=prev_layer, act=act, name=name)

        logging.info(
            "SwitchNormLayer %s: epsilon: %f act: %s" %
            (self.name, epsilon, self.act.__name__ if self.act is not None else 'No Activation')
        )

        with tf.variable_scope(name):
            x = self.inputs
            ch = x.shape[-1]
            epsilon = 1e-5

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
            ins_mean, ins_var = tf.nn.moments(x, [1, 2], keep_dims=True)
            layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

            gamma = tf.get_variable("gamma", [ch], initializer=gamma_init)
            beta = tf.get_variable("beta", [ch], initializer=beta_init)

            mean_weight_var = tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0))
            var_weight_var = tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0))

            mean_weight = tf.nn.softmax(mean_weight_var)
            var_weight = tf.nn.softmax(var_weight_var)

            mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
            var = var_weight[0] * batch_var + var_weight[1] * ins_var + var_weight[2] * layer_var

            x = (x - mean) / (tf.sqrt(var + epsilon))
            self.outputs = x * gamma + beta
            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)
        self._add_params([beta, gamma, mean_weight_var, var_weight_var])
