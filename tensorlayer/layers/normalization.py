#! /usr/bin/python
# -*- coding: utf-8 -*-

import inspect

import tensorflow as tf
from tensorflow.python.training import moving_averages

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES
from tensorlayer.layers.utils import get_collection_trainable

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'BatchNorm',
    'InstanceNorm',
    'LayerNorm',
    'LocalResponseNorm',
    'GroupNorm',
    'SwitchNorm',
]


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

    def __init__(
        self,
        decay=0.99,
        epsilon=1e-5,
        beta_init=tf.zeros_initializer,
        gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
        moving_mean_init=tf.zeros_initializer,
        moving_var_init=tf.constant_initializer(1.),
        act=None,
        name='batchnorm_layer',
    ):

        if decay > 1:
            raise ValueError("`decay` should be between 0 to 1")

        self.decay = decay
        self.epsilon = epsilon
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.act = act
        self.name = name

        for initializer in ['beta_init', 'gamma_init', 'moving_mean_init', 'moving_var_init']:
            _init = getattr(self, initializer)
            if inspect.isclass(_init):
                setattr(self, initializer, _init())

        super(BatchNorm, self).__init__()

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
            additional_str.append("is_train: %s" % self._temp_data['is_train'])
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        x_shape = self._temp_data['inputs'].get_shape()
        params_shape = x_shape[-1:]

        with tf.variable_scope(self.name):
            axis = list(range(len(x_shape) - 1))

            # 1. beta, gamma

            if self.beta_init:

                beta = self._get_tf_variable(
                    name='beta',
                    shape=params_shape,
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.beta_init,
                )

            else:
                beta = None

            if self.gamma_init:
                gamma = self._get_tf_variable(
                    name='gamma',
                    shape=params_shape,
                    dtype=self._temp_data['inputs'].dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.gamma_init,
                )
            else:
                gamma = None

            # 2.

            moving_mean = self._get_tf_variable(
                name='moving_mean',
                shape=params_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=False,
                initializer=self.moving_mean_init,
            )

            moving_variance = self._get_tf_variable(
                name='moving_variance',
                shape=params_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=False,
                initializer=self.moving_var_init,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self._temp_data['inputs'], axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, self.decay, zero_debias=False
            )  # if zero_debias=True, has bias

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if self._temp_data['is_train']:
                mean, var = mean_var_with_update()
            else:
                mean, var = moving_mean, moving_variance

            self._temp_data['outputs'] = self._apply_activation(
                tf.nn.batch_normalization(self._temp_data['inputs'], mean, var, beta, gamma, self.epsilon)
            )


class InstanceNorm(Layer):
    """The :class:`InstanceNorm` class is a for instance normalization.

    Parameters
    -----------
    act : activation function.
        The activation function of this layer.
    epsilon : float
        Eplison.
    name : str
        A unique layer name
    """

    def __init__(
        self,
        epsilon=1e-5,
        act=None,
        name='instance_norm',
    ):

        self.epsilon = epsilon
        self.act = act
        self.name = name

        super(InstanceNorm, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("epsilon: %s" % self.epsilon)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        if len(self._temp_data['inputs'].shape) not in [3, 4]:
            raise RuntimeError("`%s` only accepts input Tensor of dimension 3 or 4." % self.__class__.__name__)

        with tf.variable_scope(self.name):
            mean, var = tf.nn.moments(self._temp_data['inputs'], [1, 2], keep_dims=True)

            scale = self._get_tf_variable(
                name='scale',
                shape=[
                    self._temp_data['inputs'].get_shape()[-1],
                ],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02)
            )

            offset = self._get_tf_variable(
                name='offset',
                shape=[
                    self._temp_data['inputs'].get_shape()[-1],
                ],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.constant_initializer(0.0)
            )

            self._temp_data['outputs'] = tf.multiply(
                scale, tf.div(self._temp_data['inputs'] - mean, tf.sqrt(var + self.epsilon))
            )
            self._temp_data['outputs'] = tf.add(self._temp_data['outputs'], offset)
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])


class GroupNorm(Layer):
    """The :class:`GroupNorm` layer is for Group Normalization.
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

    def __init__(self, groups=32, epsilon=1e-06, act=None, data_format='channels_last', name='groupnorm'):
        self.groups = groups
        self.epsilon = epsilon
        self.act = act
        self.data_format = data_format
        self.name = name

        super(GroupNorm, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("groups: %s" % str(self.groups))
        except AttributeError:
            pass

        try:
            additional_str.append("epsilon: %s" % str(self.epsilon))
        except AttributeError:
            pass

        try:
            additional_str.append("data_format: %s" % self.data_format)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):
        shape = self._temp_data['inputs'].get_shape().as_list()

        if len(shape) != 4:
            raise Exception("GroupNorm only supports 2D images.")

        if self.data_format == 'channels_last':
            n_channels = shape[-1]
            int_shape = tf.concat(
                [
                    tf.shape(self._temp_data['inputs'])[0:3],
                    tf.convert_to_tensor([self.groups, n_channels // self.groups])
                ],
                axis=0
            )

        elif self.data_format == 'channels_first':
            n_channels = shape[1]
            int_shape = tf.concat(
                [
                    tf.shape(self._temp_data['inputs'])[0:1],
                    tf.convert_to_tensor([self.groups, n_channels // self.groups]),
                    tf.shape(self._temp_data['inputs'])[2:4]
                ],
                axis=0
            )

        else:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")

        if self.groups > n_channels:
            raise ValueError('Invalid groups %d for %d n_channels.' % (self.groups, n_channels))

        if n_channels % self.groups != 0:
            raise ValueError('%d n_channels is not commensurate with %d groups.' % (n_channels, self.groups))

        with tf.variable_scope(self.name):

            x = tf.reshape(self._temp_data['inputs'], int_shape)

            if self.data_format == 'channels_last':
                moments_shape = [1, 2, 4]
                weight_shape = n_channels

            else:
                moments_shape = [2, 3, 4]
                weight_shape = [1, n_channels, 1, 1]

            mean, var = tf.nn.moments(x, moments_shape, keep_dims=True)

            gamma = self._get_tf_variable(
                name='gamma',
                shape=weight_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.ones_initializer()
            )

            beta = self._get_tf_variable(
                name='beta',
                shape=weight_shape,
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.zeros_initializer()
            )

            x = (x - mean) / tf.sqrt(var + self.epsilon)

            self._temp_data['outputs'] = tf.reshape(x, tf.shape(self._temp_data['inputs'])) * gamma + beta
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])


class LayerNorm(Layer):
    """
    The :class:`LayerNorm` class is for layer normalization, see `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    """

    def __init__(
        self,
        center=True,
        scale=True,
        variables_collections=None,
        outputs_collections=None,
        begin_norm_axis=1,
        begin_params_axis=-1,
        act=None,
        name='layernorm'
    ):

        self.center = center
        self.scale = scale
        self.variables_collections = variables_collections
        self.outputs_collections = outputs_collections
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.act = act
        self.name = name

        super(LayerNorm, self).__init__()

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

        return self._str(additional_str)

    def build(self):

        with tf.variable_scope(self.name) as vs:

            self._temp_data['outputs'] = tf.contrib.layers.layer_norm(
                self._temp_data['inputs'],
                center=self.center,
                scale=self.scale,
                activation_fn=None,
                variables_collections=self.variables_collections,
                outputs_collections=self.outputs_collections,
                begin_norm_axis=self.begin_norm_axis,
                begin_params_axis=self.begin_params_axis,
                reuse=tf.get_variable_scope().reuse,
                trainable=self._temp_data['is_train'],
                scope='var',
            )

            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])

            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)


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
    name : str
        A unique layer name.
    """

    def __init__(
        self,
        depth_radius=None,
        bias=None,
        alpha=None,
        beta=None,
        name='lrn_layer',
    ):

        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.name = name

        super(LocalResponseNorm, self).__init__()

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

    def build(self):

        with tf.variable_scope(self.name):

            self._temp_data['outputs'] = tf.nn.local_response_normalization(
                self._temp_data['inputs'],
                depth_radius=self.depth_radius,
                bias=self.bias,
                alpha=self.alpha,
                beta=self.beta
            )


class SwitchNorm(Layer):
    """
    The :class:`SwitchNorm` is a switchable normalization.

    Parameters
    ----------
    act : activation function
        The activation function of this layer.
    epsilon : float
        Epsilon.
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

    def __init__(
        self,
        epsilon=1e-5,
        beta_init=tf.constant_initializer(0.0),
        gamma_init=tf.constant_initializer(1.0),
        act=None,
        name='switchnorm_layer',
    ):

        self.epsilon = epsilon
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.act = act
        self.name = name

        super(SwitchNorm, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("epsilon: %s" % self.epsilon)
        except AttributeError:
            pass

        return self._str(additional_str)

    def build(self):

        if len(self._temp_data['inputs'].shape) not in [3, 4]:
            raise RuntimeError("`%s` only accepts input Tensor of dimension 3 or 4." % self.__class__.__name__)

        with tf.variable_scope(self.name):
            ch = self._temp_data['inputs'].shape[-1]

            batch_mean, batch_var = tf.nn.moments(self._temp_data['inputs'], [0, 1, 2], keep_dims=True)
            ins_mean, ins_var = tf.nn.moments(self._temp_data['inputs'], [1, 2], keep_dims=True)
            layer_mean, layer_var = tf.nn.moments(self._temp_data['inputs'], [1, 2, 3], keep_dims=True)

            gamma = self._get_tf_variable(
                name="gamma",
                shape=[
                    ch,
                ],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.gamma_init
            )

            beta = self._get_tf_variable(
                name="beta",
                shape=[ch],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.beta_init
            )

            mean_weight_var = self._get_tf_variable(
                name="mean_weight",
                shape=[3],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.constant_initializer(1.0)
            )

            var_weight_var = self._get_tf_variable(
                name="var_weight",
                shape=[3],
                dtype=self._temp_data['inputs'].dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.constant_initializer(1.0)
            )

            mean_weight = tf.nn.softmax(mean_weight_var)
            var_weight = tf.nn.softmax(var_weight_var)

            mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
            var = var_weight[0] * batch_var + var_weight[1] * ins_var + var_weight[2] * layer_var

            x = (self._temp_data['inputs'] - mean) / (tf.sqrt(var + self.epsilon))

            self._temp_data['outputs'] = tf.add(tf.multiply(x, gamma), beta)
            self._temp_data['outputs'] = self._apply_activation(self._temp_data['outputs'])
