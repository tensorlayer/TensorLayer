# -*- coding: utf-8 -*-

import tensorflow as tf

from .. import _logging as logging
from .core import *

__all__ = [
    'LocalResponseNormLayer',
    'BatchNormLayer',
    'InstanceNormLayer',
    'LayerNormLayer',
]


class LocalResponseNormLayer(Layer):
    """The :class:`LocalResponseNormLayer` layer is for Local Response Normalization.
    See ``tf.nn.local_response_normalization`` or ``tf.nn.lrn`` for new TF version.
    The 4-D input tensor is a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.
    Within a given vector, each component is divided by the weighted square-sum of inputs within depth_radius.

    Parameters
    -----------
    layer : :class:`Layer`
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

    def __init__(
            self,
            prev_layer,
            depth_radius=None,
            bias=None,
            alpha=None,
            beta=None,
            name='lrn_layer',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info("LocalResponseNormLayer %s: depth_radius: %s, bias: %s, alpha: %s, beta: %s" % (self.name, str(depth_radius), str(bias), str(alpha),
                                                                                                     str(beta)))
        with tf.variable_scope(name):
            self.outputs = tf.nn.lrn(self.inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` is a batch normalization layer for both fully-connected and convolution outputs.
    See ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Parameters
    ----------
    layer : :class:`Layer`
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
    dtype : TensorFlow dtype
        tf.float32 (default) or tf.float16.
    name : str
        A unique layer name.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`__
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`__

    """

    def __init__(
            self,
            prev_layer,
            decay=0.9,
            epsilon=0.00001,
            act=tf.identity,
            is_train=False,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
            name='batchnorm_layer',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info("BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages

        with tf.variable_scope(name):
            axis = list(range(len(x_shape) - 1))
            # 1. beta, gamma
            variables = []
            if beta_init:
                if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                    beta_init = beta_init()
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=LayersConfig.tf_dtype, trainable=is_train)
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
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=LayersConfig.tf_dtype, trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=LayersConfig.tf_dtype,
                trainable=False,
            )

            # 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
                # logging.info("TF12 moving")
            except Exception:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                # logging.info("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act(tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

            variables.extend([moving_mean, moving_variance])

            # logging.info(len(variables))
            # for idx, v in enumerate(variables):
            #     logging.info("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)


class InstanceNormLayer(Layer):
    """The :class:`InstanceNormLayer` class is a for instance normalization.

    Parameters
    -----------
    layer : :class:`Layer`
        The previous layer.
    act : activation function.
        The activation function of this layer.
    epsilon : float
        Eplison.
    name : str
        A unique layer name

    """

    def __init__(
            self,
            prev_layer,
            act=tf.identity,
            epsilon=1e-5,
            name='instan_norm',
    ):
        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info("InstanceNormLayer %s: epsilon:%f act:%s" % (self.name, epsilon, act.__name__))

        with tf.variable_scope(name) as vs:
            mean, var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)
            scale = tf.get_variable(
                'scale', [self.inputs.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=LayersConfig.tf_dtype)
            offset = tf.get_variable('offset', [self.inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0), dtype=LayersConfig.tf_dtype)
            self.outputs = scale * tf.div(self.inputs - mean, tf.sqrt(var + epsilon)) + offset
            self.outputs = act(self.outputs)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)


class LayerNormLayer(Layer):
    """
    The :class:`LayerNormLayer` class is for layer normalization, see `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    Parameters
    ----------
    layer : :class:`Layer`
        The previous layer.
    act : activation function
        The activation function of this layer.
    others : _
        `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`__.

    """

    def __init__(self,
                 prev_layer,
                 center=True,
                 scale=True,
                 act=tf.identity,
                 reuse=None,
                 variables_collections=None,
                 outputs_collections=None,
                 trainable=True,
                 begin_norm_axis=1,
                 begin_params_axis=-1,
                 name='layernorm'):

        Layer.__init__(self, prev_layer=prev_layer, name=name)
        self.inputs = prev_layer.outputs
        logging.info("LayerNormLayer %s: act:%s" % (self.name, act.__name__))

        if tf.__version__ < "1.3":
            # raise Exception("Please use TF 1.3+")
            with tf.variable_scope(name) as vs:
                self.outputs = tf.contrib.layers.layer_norm(
                    self.inputs,
                    center=center,
                    scale=scale,
                    activation_fn=act,
                    reuse=reuse,
                    variables_collections=variables_collections,
                    outputs_collections=outputs_collections,
                    trainable=trainable,
                    # begin_norm_axis=begin_norm_axis,
                    # begin_params_axis=begin_params_axis,
                    scope='var',
                )
                variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
        else:
            with tf.variable_scope(name) as vs:
                self.outputs = tf.contrib.layers.layer_norm(
                    self.inputs,
                    center=center,
                    scale=scale,
                    activation_fn=act,
                    reuse=reuse,
                    variables_collections=variables_collections,
                    outputs_collections=outputs_collections,
                    trainable=trainable,
                    begin_norm_axis=begin_norm_axis,
                    begin_params_axis=begin_params_axis,
                    scope='var',
                )
                variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # self.all_layers = list(layer.all_layers)
        # self.all_params = list(layer.all_params)
        # self.all_drop = dict(layer.all_drop)
        self.all_layers.append(self.outputs)
        self.all_params.extend(variables)
