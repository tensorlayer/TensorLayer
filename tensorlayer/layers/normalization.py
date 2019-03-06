#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.training import moving_averages
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorlayer.layers.core import Layer
# from tensorlayer.layers.core import LayersConfig
# from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES
from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'LocalResponseNorm',
    'BatchNorm', # FIXME: wthether to keep BatchNorm
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'InstanceNorm',
    'LayerNorm',
    'GroupNorm',
    'SwitchNorm',
]


class LocalResponseNorm(Layer):
    """The :class:`LocalResponseNorm` layer is for Local Response Normalization.
    See ``tf.nn.local_response_normalization`` or ``tf.nn.lrn`` for new TF version.
    The 4-D input tensor is a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.
    Within a given vector, each component is divided by the weighted square-sum of inputs within depth_radius.

    Parameters
    -----------
    depth_radius : int
        Depth radius. 0-D. Half-width of the 1-D normalization window.
    bias : float
        An offset which is usually positive and shall avoid dividing by 0.
    alpha : float
        A scale factor which is usually positive.
    beta : float
        An exponent.
    name : None or str
        A unique layer name.

    """

    def __init__(
            self,
            depth_radius=None,
            bias=None,
            alpha=None,
            beta=None,
            name=None,  #'lrn',
    ):
        # super(LocalResponseNorm, self).__init__(prev_layer=prev_layer, name=name)
        super().__init__(name)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        logging.info(
            "LocalResponseNorm %s: depth_radius: %s, bias: %s, alpha: %s, beta: %s" %
            (self.name, str(depth_radius), str(bias), str(alpha), str(beta))
        )

    def build(self, inputs):
        pass

    def forward(self, inputs):
        """
        prev_layer : :class:`Layer`
            The previous layer with a 4D output shape.
        """
        outputs = tf.nn.lrn(inputs, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)
        return outputs


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


class BatchNorm(Layer):
    """
    The :class:`BatchNorm` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
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
    moving_mean_init : initializer or None
        The initializer for initializing moving mean, if None, skip moving mean.
    moving_var_init : initializer or None
        The initializer for initializing moving var, if None, skip moving var.
    num_features: int
        Number of features for input tensor. Useful to build layer if using BatchNorm1d, BatchNorm2d or BatchNorm3d,
        but should be left as None if using BatchNorm.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    def __init__(
            self,
            decay=0.9,
            epsilon=0.00001,
            act=None,
            is_train=False,
            beta_init=tl.initializers.zeros(),
            gamma_init=tl.initializers.random_normal(mean=1.0, stddev=0.002),
            moving_mean_init=tl.initializers.zeros(),
            moving_var_init=tl.initializers.zeros(),
            # beta_init=tf.compat.v1.initializers.zeros(),
            # gamma_init=tf.compat.v1.initializers.random_normal(mean=1.0, stddev=0.002),
            # moving_mean_init=tf.compat.v1.initializers.zeros(),
            # moving_var_init=tf.compat.v1.initializers.zeros(),
            num_features=None,
            data_format='channels_last',
            name=None,
    ):
        super(BatchNorm, self).__init__(name=name)
        self.act = act
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.num_features = num_features

        if num_features is not None:
            if not isinstance(self, BatchNorm1d) and not isinstance(self, BatchNorm2d) and not isinstance(self, BatchNorm3d):
                raise ValueError("Please use BatchNorm1d or BatchNorm2d or BatchNorm3d instead of BatchNorm "
                                 "if you want to specify 'num_features'.")
            self.build(None)
            self._built = True

        logging.info(
            "BatchNorm %s: decay: %f epsilon: %f act: %s is_train: %s" %
            (self.name, decay, epsilon, self.act.__name__ if self.act is not None else 'No Activation', is_train)
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(num_features={num_features}, decay={decay}'
             ', epsilon={epsilon}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = len(inputs_shape) - 1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [1] * len(inputs_shape)
        params_shape[axis] = channels

        axes = [i for i in range(len(inputs_shape)) if i != axis]
        return params_shape, axes

    def build(self, inputs_shape):
        if self.decay < 0 or 1 < self.decay:
            raise Exception("decay should be between 0 to 1")

        # x_shape = self.inputs.get_shape()
        # if self.data_format == 'channels_last':
        #     axis = len(inputs_shape) - 1
        #     channels = inputs_shape[-1]
        #     params_shape = [1] * (len(inputs_shape) - 1) + [channels]
        # elif self.data_format == 'channels_first':
        #     axis = 1
        #     channels = inputs_shape[1]
        #     params_shape = [1, channels] + [1] * (len(inputs_shape) - 2)
        # else:
        #     raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))
        #
        # # params_shape = inputs_shape[axis]
        # self.axes = [i for i in range(len(inputs_shape)) if i != axis]

        params_shape, self.axes = self._get_param_shape(inputs_shape)

        self.beta, self.gamma = None, None
        if self.beta_init:
            self.beta = self._get_weights("beta", shape=params_shape, init=self.beta_init)
        # with tf.variable_scope(name):
        #     axes = [i for i in range(len(x_shape)) if i != axis]
        #
        #     # 1. beta, gamma
        #     variables = []
        #
        #     if beta_init:
        #
        #         if beta_init == tf.zeros_initializer:
        #             beta_init = beta_init()
        #
        #         beta = tf.get_variable(
        #             'beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype, trainable=is_train
        #         )
        #
        #         variables.append(beta)
        #
        #     else:
        #         beta = None
        if self.gamma_init:
            self.gamma = self._get_weights("gamma", shape=params_shape, init=self.gamma_init)
        #     if gamma_init:
        #         gamma = tf.get_variable(
        #             'gamma',
        #             shape=params_shape,
        #             initializer=gamma_init,
        #             dtype=LayersConfig.tf_dtype,
        #             trainable=is_train,
        #         )
        #         variables.append(gamma)
        #     else:
        #         gamma = None
        #
        #     # 2.
        self.moving_mean = self._get_weights("moving_mean", shape=params_shape, init=self.moving_mean_init)
        #     moving_mean = tf.get_variable(
        #         'moving_mean', params_shape, initializer=moving_mean_init, dtype=LayersConfig.tf_dtype, trainable=False
        #     )
        #
        #     moving_variance = tf.get_variable(
        #         'moving_variance',
        #         params_shape,
        #         initializer=tf.constant_initializer(1.),
        #         dtype=LayersConfig.tf_dtype,
        #         trainable=False,
        #     )
        self.moving_var = self._get_weights("moving_var", shape=params_shape, init=self.moving_var_init)

    def forward(self, inputs):
        mean, var = tf.nn.moments(inputs, self.axes)
        if self.is_train:
            # update moving_mean and moving_var
            self.moving_mean = moving_averages.assign_moving_average(self.moving_mean, mean,
                                                                       self.decay, zero_debias=False)
            self.moving_var = moving_averages.assign_moving_average(self.moving_var, var,
                                                                      self.decay, zero_debias=False)
            outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma,
                                          self.epsilon, self.data_format)
        else:
            outputs = batch_normalization(inputs, self.moving_mean, self.moving_var, self.beta, self.gamma,
                                          self.epsilon, self.data_format)
        if self.act:
            outputs = self.act(outputs)
        return outputs
        #     # 3.
        #     # These ops will only be preformed when training.
        #     mean, variance = tf.nn.moments(self.inputs, axes)
        #     update_moving_mean = moving_averages.assign_moving_average(
        #         moving_mean, mean, decay, zero_debias=False
        #     )  # if zero_debias=True, has bias
        #     update_moving_variance = moving_averages.assign_moving_average(
        #         moving_variance, variance, decay, zero_debias=False
        #     )  # if zero_debias=True, has bias
        #
        #     def mean_var_with_update():
        #         with tf.control_dependencies([update_moving_mean, update_moving_variance]):
        #             return tf.identity(mean), tf.identity(variance)
        #
        #     if is_train:
        #         mean, var = mean_var_with_update()
        #     else:
        #         mean, var = moving_mean, moving_variance
        #
        #     self.outputs = self._apply_activation(
        #         batch_normalization(self.inputs, mean, var, beta, gamma, epsilon, data_format)
        #     )
        #
        #     variables.extend([moving_mean, moving_variance])
        #
        # self._add_layers(self.outputs)
        # self._add_params(variables)


class BatchNorm1d(BatchNorm):
    # TODO: documentation pending, need test
    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = 2
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        if self.num_features is None:
            channels = inputs_shape[axis]
        else:
            channels = self.num_features
        params_shape = [1] * 3
        params_shape[axis] = channels

        axes = [i for i in range(3) if i != axis]
        return params_shape, axes


class BatchNorm2d(BatchNorm):
    # TODO: documentation pending
    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = 3
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        if self.num_features is None:
            channels = inputs_shape[axis]
        else:
            channels = self.num_features
        params_shape = [1] * 4
        params_shape[axis] = channels

        axes = [i for i in range(4) if i != axis]
        return params_shape, axes


class BatchNorm3d(BatchNorm):
    # TODO: documentation pending, need test
    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = 4
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        if self.num_features is None:
            channels = inputs_shape[axis]
        else:
            channels = self.num_features
        params_shape = [1] * 5
        params_shape[axis] = channels

        axes = [i for i in range(5) if i != axis]
        return params_shape, axes


class InstanceNorm(Layer):
    """The :class:`InstanceNorm` class is a for instance normalization.

    Parameters
    -----------
    act : activation function.
        The activation function of this layer.
    epsilon : float
        Eplison.
    name : None or str
        A unique layer name

    """

    def __init__(
            self,
            act=None,
            epsilon=1e-5,
            name=None,  #'instan_norm',
    ):
        # super(InstanceNorm, self).__init__(prev_layer=prev_layer, act=act, name=name)
        super().__init__(name)
        self.act = act
        self.epsilon = epsilon

        logging.info(
            "InstanceNorm %s: epsilon: %f act: %s" %
            (self.name, epsilon, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):
        # self.scale = tf.compat.v1.get_variable(
        #     self.name + '\scale', [inputs.get_shape()[-1]],
        #     initializer=tf.compat.v1.initializers.truncated_normal(mean=1.0, stddev=0.02), dtype=LayersConfig.tf_dtype
        # )
        self.scale = self._get_weights(
            "scale", shape=[inputs_shape[-1]], init=tf.compat.v1.initializers.truncated_normal(mean=1.0, stddev=0.02)
        )
        # self.offset = tf.compat.v1.get_variable(
        #     self.name + '\offset', [inputs.get_shape()[-1]], initializer=tf.compat.v1.initializers.constant(0.0),
        #     dtype=LayersConfig.tf_dtype
        # )
        self.offset = self._get_weights(
            "offset", shape=[inputs_shape[-1]], init=tf.compat.v1.initializers.constant(0.0)
        )
        # self.add_weights([self.scale, self.offset])

    def forward(self, inputs):

        mean, var = tf.nn.moments(x=inputs, axes=[1, 2], keepdims=True)

        outputs = self.scale * tf.compat.v1.div(inputs - mean, tf.sqrt(var + self.epsilon)) + self.offset
        outputs = self.act(outputs)

        return outputs

        # with tf.variable_scope(name) as vs:
        #     mean, var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)
        #
        #     scale = tf.get_variable(
        #         'scale', [self.inputs.get_shape()[-1]],
        #         initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=LayersConfig.tf_dtype
        #     )
        #
        #     offset = tf.get_variable(
        #         'offset', [self.inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0),
        #         dtype=LayersConfig.tf_dtype
        #     )
        #
        #     self.outputs = scale * tf.div(self.inputs - mean, tf.sqrt(var + epsilon)) + offset
        #     self.outputs = self._apply_activation(self.outputs)
        #
        #     variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        #
        # self._add_layers(self.outputs)
        # self._add_params(variables)


# FIXME : not sure about the correctness, need testing
class LayerNorm(Layer):
    """
    The :class:`LayerNorm` class is for layer normalization, see `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        The previous layer.
    act : activation function
        The activation function of this layer.
    others : _
        `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    """

    def __init__(
            self,  #prev_layer,
            center=True,
            scale=True,
            act=None,
            # reuse=None,
            # variables_collections=None,
            # outputs_collections=None,
            # trainable=True,
            epsilon=1e-12,
            begin_norm_axis=1,
            begin_params_axis=-1,
            beta_init=tl.initializers.zeros(),
            gamma_init=tl.initializers.ones(),
            data_format='channels_last',
            name=None,
    ):

        # super(LayerNorm, self).__init__(prev_layer=prev_layer, act=act, name=name)
        super(LayerNorm, self).__init__(name)
        self.center = center
        self.scale = scale
        self.act = act
        self.epsilon = epsilon
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.data_format = data_format

        logging.info(
            "LayerNorm %s: act: %s" % (self.name, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):
        params_shape = inputs_shape[self.begin_params_axis:]
        self.beta, self.gamma = None, None
        if self.center:
            self.beta = self._get_weights("beta", shape=params_shape, init=self.beta_init)
        if self.scale:
            self.gamma = self._get_weights("gamma", shape=params_shape, init=self.gamma_init)

        self.norm_axes = range(self.begin_norm_axis, len(inputs_shape))

    def forward(self, inputs):
        mean, var = tf.nn.moments(inputs, self.norm_axes, keepdims=True)
        # compute layer normalization using batch_normalization function
        outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma,
                                      self.epsilon, data_format=self.data_format)
        if self.act:
            outputs = self.act(outputs)
        return outputs
    #     with tf.compat.v1.variable_scope(name) as vs:
    #         self.outputs = tf.contrib.layers.layer_norm(
    #             self.inputs,
    #             center=center,
    #             scale=scale,
    #             activation_fn=self.act,
    #             reuse=reuse,
    #             variables_collections=variables_collections,
    #             outputs_collections=outputs_collections,
    #             trainable=trainable,
    #             begin_norm_axis=begin_norm_axis,
    #             begin_params_axis=begin_params_axis,
    #             scope='var',
    #         )
    #
    #         variables = tf.compat.v1.get_collection("TF_GRAPHKEYS_VARIABLES", scope=vs.name)
    #
    #     self._add_layers(self.outputs)
    #     self._add_params(variables)


class GroupNorm(Layer):
    """The :class:`GroupNorm` layer is for Group Normalization.
    See `tf.contrib.layers.group_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/group_norm>`__.

    Parameters
    -----------
    # prev_layer : :class:`Layer`
    #     The previous layer.
    groups : int
        The number of groups
    act : activation function
        The activation function of this layer.
    epsilon : float
        Eplison.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name

    """

    def __init__(self, groups=32, epsilon=1e-06, act=None, data_format='channels_last', name=None):  #'groupnorm'):
        # super(GroupNorm, self).__init__(prev_layer=prev_layer, act=act, name=name)
        super().__init__(name)
        self.groups = groups
        self.epsilon = epsilon
        self.act = act
        self.data_format = data_format

        logging.info(
            "GroupNorm %s: act: %s" % (self.name, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):
        # shape = inputs.get_shape().as_list()
        if len(inputs_shape) != 4:
            raise Exception("This GroupNorm only supports 2D images.")

        if self.data_format == 'channels_last':
            channels = inputs_shape[-1]
            self.int_shape = tf.concat(
                [#tf.shape(input=self.inputs)[0:3],
                inputs_shape[0:3],
                tf.convert_to_tensor(value=[self.groups, channels // self.groups])], axis=0
            )
        elif self.data_format == 'channels_first':
            channels = inputs_shape[1]
            self.int_shape = tf.concat(
                [
                    # tf.shape(input=self.inputs)[0:1],
                    inputs_shape[0:1],
                    tf.convert_to_tensor(value=[self.groups, channels // self.groups]),
                    # tf.shape(input=self.inputs)[2:4]
                    inputs_shape[2:4],
                ],
                axis=0
            )
        else:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")

        if self.groups > channels:
            raise ValueError('Invalid groups %d for %d channels.' % (self.groups, channels))
        if channels % self.groups != 0:
            raise ValueError('%d channels is not commensurate with %d groups.' % (channels, self.groups))

        if self.data_format == 'channels_last':
            # mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
            self.gamma = self._get_weights("gamma", shape=channels, init=tl.initializers.ones())
            # self.gamma = tf.compat.v1.get_variable('gamma', channels, initializer=tf.compat.v1.initializers.ones())
            self.beta = self._get_weights("beta", shape=channels, init=tl.initializers.zeros())
            # self.beta = tf.compat.v1.get_variable('beta', channels, initializer=tf.compat.v1.initializers.zeros())
        elif self.data_format == 'channels_first':
            # mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            self.gamma = self._get_weights("gamma", shape=[1, channels, 1, 1], init=tl.initializers.ones())
            # self.gamma = tf.compat.v1.get_variable('gamma', [1, channels, 1, 1], initializer=tf.compat.v1.initializers.ones())
            self.beta = self._get_weights("beta", shape=[1, channels, 1, 1], init=tl.initializers.zeros())
            # self.beta = tf.compat.v1.get_variable('beta', [1, channels, 1, 1], initializer=tf.compat.v1.initializers.zeros())
        # self.add_weights([self.gamma, self.bata])

    def forward(self, inputs):
        x = tf.reshape(inputs, self.int_shape)
        if self.data_format == 'channels_last':
            mean, var = tf.nn.moments(x=x, axes=[1, 2, 4], keepdims=True)
        elif self.data_format == 'channels_first':
            mean, var = tf.nn.moments(x=x, axes=[2, 3, 4], keepdims=True)
        else:
            raise Exception("unknown data_format")
        x = (x - mean) / tf.sqrt(var + self.epsilon)

        outputs = tf.reshape(x, tf.shape(input=inputs)) * self.gamma + self.beta
        if self.act:
            outputs = self.act(outputs)
        return outputs


class SwitchNorm(Layer):
    """
    The :class:`SwitchNorm` is a switchable normalization.

    Parameters
    ----------
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
    moving_mean_init : initializer or None
        The initializer for initializing moving mean, if None, skip moving mean.
    data_format : str
        channels_last 'channel_last' (default) or channels_first.
    name : None or str
        A unique layer name.

    References
    ----------
    - `Differentiable Learning-to-Normalize via Switchable Normalization <https://arxiv.org/abs/1806.10779>`__
    - `Zhihu (CN) <https://zhuanlan.zhihu.com/p/39296570?utm_source=wechat_session&utm_medium=social&utm_oi=984862267107651584>`__

    """

    def __init__(
            self,
            act=None,
            epsilon=1e-5,
            beta_init=tl.initializers.constant(0.0),
            gamma_init=tl.initializers.constant(1.0),
            moving_mean_init=tl.initializers.zeros(),
            # beta_init=tf.compat.v1.initializers.constant(0.0),
            # gamma_init=tf.compat.v1.initializers.constant(1.0),
            # moving_mean_init=tf.compat.v1.initializers.zeros(),
            data_format='channels_last',
            name=None,  #'switchnorm',
    ):
        # super(SwitchNorm, self).__init__(prev_layer=prev_layer, act=act, name=name)
        super().__init__(name)
        self.act = act
        self.epsilon = epsilon
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.data_format = data_format

        logging.info(
            "SwitchNorm %s: epsilon: %f act: %s" %
            (self.name, epsilon, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def build(self, inputs_shape):
        if len(inputs_shape) != 4:
            raise Exception("This SwitchNorm only supports 2D images.")
        if self.data_format != 'channels_last':
            raise Exception("This SwitchNorm only supports channels_last.")
        ch = inputs_shape[-1]
        self.gamma = self._get_weights("gamma", shape=[ch], init=self.gamma_init)
        # self.gamma = tf.compat.v1.get_variable("gamma", [ch], initializer=gamma_init)
        self.beta = self._get_weights("beta", shape=[ch], init=self.beta_init)
        # self.beta = tf.compat.v1.get_variable("beta", [ch], initializer=beta_init)

        self.mean_weight_var = self._get_weights("mean_weight", shape=[3], init=tl.initializers.constant(1.0))
        # self.mean_weight_var = tf.compat.v1.get_variable("mean_weight", [3], initializer=tf.compat.v1.initializers.constant(1.0))
        self.var_weight_var = self._get_weights("var_weight", shape=[3], init=tl.initializers.constant(1.0))
        # self.var_weight_var = tf.compat.v1.get_variable("var_weight", [3], initializer=tf.compat.v1.initializers.constant(1.0))

        # self.add_weights([self.gamma, self.beta, self.mean_weight_var, self.var_weight_var])

    def forward(self, inputs):

        batch_mean, batch_var = tf.nn.moments(x=inputs, axes=[0, 1, 2], keepdims=True)
        ins_mean, ins_var = tf.nn.moments(x=inputs, axes=[1, 2], keepdims=True)
        layer_mean, layer_var = tf.nn.moments(x=inputs, axes=[1, 2, 3], keepdims=True)

        mean_weight = tf.nn.softmax(self.mean_weight_var)
        var_weight = tf.nn.softmax(self.var_weight_var)

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_weight[0] * batch_var + var_weight[1] * ins_var + var_weight[2] * layer_var

        inputs = (inputs - mean) / (tf.sqrt(var + self.epsilon))
        outputs = inputs * self.gamma + self.beta
        if self.act:
            outputs = self.act(outputs)
        return outputs
