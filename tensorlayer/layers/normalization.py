#! /usr/bin/python
# -*- coding: utf-8 -*-

import copy
import inspect
import random
import time
import warnings

import numpy as np
import tensorflow as tf
from six.moves import xrange

from . import cost, files, iterate, ops, utils, visualize
from .core import *

# ## Normalization layer
class LocalResponseNormLayer(Layer):
    """The :class:`LocalResponseNormLayer` class is for Local Response Normalization, see ``tf.nn.local_response_normalization`` or ``tf.nn.lrn`` for new TF version.
    The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently.
    Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius.

    Parameters
    -----------
    layer : a layer class. Must be one of the following types: float32, half. 4-D.
    depth_radius : An optional int. Defaults to 5. 0-D. Half-width of the 1-D normalization window.
    bias : An optional float. Defaults to 1. An offset (usually positive to avoid dividing by 0).
    alpha : An optional float. Defaults to 1. A scale factor, usually positive.
    beta : An optional float. Defaults to 0.5. An exponent.
    name : A string or None, an optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            depth_radius=None,
            bias=None,
            alpha=None,
            beta=None,
            name='lrn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] LocalResponseNormLayer %s: depth_radius: %d, bias: %f, alpha: %f, beta: %f" % (self.name, depth_radius, bias, alpha, beta))
        with tf.variable_scope(name) as vs:
            self.outputs = tf.nn.lrn(self.inputs, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    dtype : tf.float32 (default) or tf.float16
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """

    def __init__(
            self,
            layer=None,
            decay=0.9,
            epsilon=0.00001,
            act=tf.identity,
            is_train=False,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),  # tf.ones_initializer,
            # dtype = tf.float32,
            name='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=D_TYPE, trainable=is_train)  #, restore=restore)

            gamma = tf.get_variable(
                'gamma',
                shape=params_shape,
                initializer=gamma_init,
                dtype=D_TYPE,
                trainable=is_train,
            )  #restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=D_TYPE, trainable=False)  #   restore=restore)
            moving_variance = tf.get_variable(
                'moving_variance',
                params_shape,
                initializer=tf.constant_initializer(1.),
                dtype=D_TYPE,
                trainable=False,
            )  #   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            else:
                self.outputs = act(tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

            variables = [beta, gamma, moving_mean, moving_variance]

            # print(len(variables))
            # for idx, v in enumerate(variables):
            #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


# class BatchNormLayer_TF(Layer):   # Work well TF contrib https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     center: If True, subtract `beta`. If False, `beta` is ignored.
#     scale: If True, multiply by `gamma`. If False, `gamma` is
#         not used. When the next layer is linear (also e.g. `nn.relu`), this can be
#         disabled since the scaling can be done by the next layer.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.95,#.999,
#         center = True,
#         scale = True,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = False,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         from tensorflow.contrib.layers.python.layers import utils
#         from tensorflow.contrib.framework.python.ops import variables
#         from tensorflow.python.ops import init_ops
#         from tensorflow.python.ops import nn
#         from tensorflow.python.training import moving_averages
#         from tensorflow.python.framework import ops
#         from tensorflow.python.ops import variable_scope
#         variables_collections = None
#         outputs_collections=None
#         updates_collections=None#ops.GraphKeys.UPDATE_OPS
#         # with variable_scope.variable_op_scope([inputs],
#         #                                     scope, 'BatchNorm', reuse=reuse) as sc:
#         # with variable_scope.variable_op_scope([self.inputs], None, name) as vs:
#         with tf.variable_scope(name) as vs:
#             inputs_shape = self.inputs.get_shape()
#             dtype = self.inputs.dtype.base_dtype
#             axis = list(range(len(inputs_shape) - 1)) # [0, 1, 2]
#             params_shape = inputs_shape[-1:]
#             # Allocate parameters for the beta and gamma of the normalization.
#             beta, gamma = None, None
#             if center:
#               beta_collections = utils.get_variable_collections(variables_collections,
#                                                                 'beta')
#               beta = variables.model_variable('beta',
#                                               shape=params_shape,
#                                               dtype=dtype,
#                                             #   initializer=init_ops.zeros_initializer,
#                                               initializer=beta_init,
#                                               collections=beta_collections,)
#                                             #   trainable=trainable)
#             if scale:
#               gamma_collections = utils.get_variable_collections(variables_collections,
#                                                                  'gamma')
#               gamma = variables.model_variable('gamma',
#                                                shape=params_shape,
#                                                dtype=dtype,
#                                             #    initializer=init_ops.ones_initializer,
#                                                initializer=gamma_init,
#                                                collections=gamma_collections,)
#                                             #    trainable=trainable)
#             # Create moving_mean and moving_variance variables and add them to the
#             # appropiate collections.
#             moving_mean_collections = utils.get_variable_collections(
#                 variables_collections,
#                 'moving_mean')
#             moving_mean = variables.model_variable(
#                 'moving_mean',
#                 shape=params_shape,
#                 dtype=dtype,
#                 # initializer=init_ops.zeros_initializer,
#                 initializer=tf.zeros_initializer,
#                 trainable=False,
#                 collections=moving_mean_collections)
#             moving_variance_collections = utils.get_variable_collections(
#                 variables_collections,
#                 'moving_variance')
#             moving_variance = variables.model_variable(
#                 'moving_variance',
#                 shape=params_shape,
#                 dtype=dtype,
#                 # initializer=init_ops.ones_initializer,
#                 initializer=tf.constant_initializer(1.),
#                 trainable=False,
#                 collections=moving_variance_collections)
#             if is_train:
#               # Calculate the moments based on the individual batch.
#               mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
#               # Update the moving_mean and moving_variance moments.
#             #   update_moving_mean = moving_averages.assign_moving_average(
#             #       moving_mean, mean, decay)
#             #   update_moving_variance = moving_averages.assign_moving_average(
#             #       moving_variance, variance, decay)
#             #   if updates_collections is None:
#             #     # Make sure the updates are computed here.
#             #       with ops.control_dependencies([update_moving_mean,
#             #                                        update_moving_variance]):
#             #          outputs = nn.batch_normalization(
#             #               self.inputs, mean, variance, beta, gamma, epsilon)
#
#               update_moving_mean = tf.assign(moving_mean,
#                                    moving_mean * decay + mean * (1 - decay))
#               update_moving_variance = tf.assign(moving_variance,
#                                   moving_variance * decay + variance * (1 - decay))
#               with tf.control_dependencies([update_moving_mean, update_moving_variance]):
#                   outputs = nn.batch_normalization(
#                               self.inputs, mean, variance, beta, gamma, epsilon)
#             #   else:
#             #     # Collect the updates to be computed later.
#             #     ops.add_to_collections(updates_collections, update_moving_mean)
#             #     ops.add_to_collections(updates_collections, update_moving_variance)
#             #     outputs = nn.batch_normalization(
#             #         self.inputs, mean, variance, beta, gamma, epsilon)
#             else:
#             #   mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
#               outputs = nn.batch_normalization(
#                   self.inputs, moving_mean, moving_variance, beta, gamma, epsilon)
#                 # self.inputs, mean, variance, beta, gamma, epsilon)
#             outputs.set_shape(self.inputs.get_shape())
#             # if activation_fn:
#             self.outputs = act(outputs)
#
#             # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
#             # return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
#             variables = [beta, gamma, moving_mean, moving_variance]
#
#         mean, variance = nn.moments(self.inputs, axis, shift=moving_mean)
#         self.check_mean = mean
#         self.check_variance = variance
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )
#
# class BatchNormLayer5(Layer):   # Akara Work well
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.9,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = False,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         x_shape = self.inputs.get_shape()
#         params_shape = x_shape[-1:]
#
#         from tensorflow.python.training import moving_averages
#         from tensorflow.python.ops import control_flow_ops
#
#         with tf.variable_scope(name) as vs:
#             axis = list(range(len(x_shape) - 1))
#
#             ## 1. beta, gamma
#             beta = tf.get_variable('beta', shape=params_shape,
#                                initializer=beta_init,
#                                trainable=is_train)#, restore=restore)
#
#             gamma = tf.get_variable('gamma', shape=params_shape,
#                                 initializer=gamma_init, trainable=is_train,
#                                 )#restore=restore)
#
#             ## 2. moving variables during training (not update by gradient!)
#             moving_mean = tf.get_variable('moving_mean',
#                                       params_shape,
#                                       initializer=tf.zeros_initializer,
#                                       trainable=False,)#   restore=restore)
#             moving_variance = tf.get_variable('moving_variance',
#                                           params_shape,
#                                           initializer=tf.constant_initializer(1.),
#                                           trainable=False,)#   restore=restore)
#
#             batch_mean, batch_var = tf.nn.moments(self.inputs, axis)
#             ## 3.
#             # These ops will only be preformed when training.
#             def mean_var_with_update():
#                 try:    # TF12
#                     update_moving_mean = moving_averages.assign_moving_average(
#                                     moving_mean, batch_mean, decay, zero_debias=False)     # if zero_debias=True, has bias
#                     update_moving_variance = moving_averages.assign_moving_average(
#                                     moving_variance, batch_var, decay, zero_debias=False) # if zero_debias=True, has bias
#                     # print("TF12 moving")
#                 except Exception as e:  # TF11
#                     update_moving_mean = moving_averages.assign_moving_average(
#                                     moving_mean, batch_mean, decay)
#                     update_moving_variance = moving_averages.assign_moving_average(
#                                     moving_variance, batch_var, decay)
#                     # print("TF11 moving")
#
#             # def mean_var_with_update():
#                 with tf.control_dependencies([update_moving_mean, update_moving_variance]):
#                     # return tf.identity(update_moving_mean), tf.identity(update_moving_variance)
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#
#             # if not is_train:
#             if is_train:
#                 mean, var = mean_var_with_update()
#             else:
#                 mean, var = (moving_mean, moving_variance)
#
#             normed = tf.nn.batch_normalization(
#               x=self.inputs,
#               mean=mean,
#               variance=var,
#               offset=beta,
#               scale=gamma,
#               variance_epsilon=epsilon,
#               name="tf_bn"
#             )
#             self.outputs = act( normed )
#
#             variables = [beta, gamma, moving_mean, moving_variance]
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
#             # exit()
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )
#         # self.all_params.extend( [beta, gamma] )
#
# class BatchNormLayer4(Layer): # work TFlearn https://github.com/tflearn/tflearn/blob/master/tflearn/layers/normalization.py
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         input_shape = self.inputs.get_shape()
#         # params_shape = input_shape[-1:]
#         input_ndim = len(input_shape)
#         from tensorflow.python.training import moving_averages
#         from tensorflow.python.ops import control_flow_ops
#
#         # gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)
#
#         # Variable Scope fix for older TF
#         scope = name
#         try:
#             vscope = tf.variable_scope(scope, default_name=name, values=[self.inputs],)
#                                     #    reuse=reuse)
#         except Exception:
#             vscope = tf.variable_op_scope([self.inputs], scope, name)#, reuse=reuse)
#
#         with vscope as scope:
#             name = scope.name
#         # with tf.variable_scope(name) as vs:
#             beta = tf.get_variable('beta', shape=[input_shape[-1]],
#                                 initializer=beta_init,)
#                             #    initializer=tf.constant_initializer(beta),)
#                             #    trainable=trainable, )#restore=restore)
#             gamma = tf.get_variable('gamma', shape=[input_shape[-1]],
#                                 initializer=gamma_init, )#trainable=trainable,)
#                                 # restore=restore)
#
#             axis = list(range(input_ndim - 1))
#             moving_mean = tf.get_variable('moving_mean',
#                                       input_shape[-1:],
#                                       initializer=tf.zeros_initializer,
#                                       trainable=False,)
#                                     #   restore=restore)
#             moving_variance = tf.get_variable('moving_variance',
#                                           input_shape[-1:],
#                                           initializer=tf.constant_initializer(1.),
#                                           trainable=False,)
#                                         #   restore=restore)
#
#             # Define a function to update mean and variance
#             def update_mean_var():
#                 mean, variance = tf.nn.moments(self.inputs, axis)
#
#                 # Fix TF 0.12
#                 try:
#                     update_moving_mean = moving_averages.assign_moving_average(
#                         moving_mean, mean, decay, zero_debias=False)            # if zero_debias=True, accuracy is high ..
#                     update_moving_variance = moving_averages.assign_moving_average(
#                         moving_variance, variance, decay, zero_debias=False)
#                 except Exception as e:  # TF 11
#                     update_moving_mean = moving_averages.assign_moving_average(
#                         moving_mean, mean, decay)
#                     update_moving_variance = moving_averages.assign_moving_average(
#                         moving_variance, variance, decay)
#
#                 with tf.control_dependencies(
#                         [update_moving_mean, update_moving_variance]):
#                     return tf.identity(mean), tf.identity(variance)
#
#             # Retrieve variable managing training mode
#             # is_training = tflearn.get_training_mode()
#             if not is_train:    # test : mean=0, std=1
#             # if is_train:      # train : mean=0, std=1
#                 is_training = tf.cast(tf.ones([]), tf.bool)
#             else:
#                 is_training = tf.cast(tf.zeros([]), tf.bool)
#             mean, var = tf.cond(
#                 is_training, update_mean_var, lambda: (moving_mean, moving_variance))
#                             #  ones                 zeros
#             try:
#                 inference = tf.nn.batch_normalization(
#                     self.inputs, mean, var, beta, gamma, epsilon)
#                 inference.set_shape(input_shape)
#             # Fix for old Tensorflow
#             except Exception as e:
#                 inference = tf.nn.batch_norm_with_global_normalization(
#                     self.inputs, mean, var, beta, gamma, epsilon,
#                     scale_after_normalization=True,
#                 )
#                 inference.set_shape(input_shape)
#
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         # Add attributes for easy access
#         # inference.scope = scope
#         inference.scope = name
#         inference.beta = beta
#         inference.gamma = gamma
#
#         self.outputs = act( inference )
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )

# class BatchNormLayer2(Layer):   # don't work http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         x_shape = self.inputs.get_shape()
#         params_shape = x_shape[-1:]
#
#         with tf.variable_scope(name) as vs:
#             gamma = tf.get_variable("gamma", shape=params_shape,
#                         initializer=gamma_init)
#             beta = tf.get_variable("beta", shape=params_shape,
#                         initializer=beta_init)
#             pop_mean = tf.get_variable("pop_mean", shape=params_shape,
#                         initializer=tf.zeros_initializer, trainable=False)
#             pop_var = tf.get_variable("pop_var", shape=params_shape,
#                         initializer=tf.constant_initializer(1.), trainable=False)
#
#             if is_train:
#                 batch_mean, batch_var = tf.nn.moments(self.inputs, list(range(len(x_shape) - 1)))
#                 train_mean = tf.assign(pop_mean,
#                                        pop_mean * decay + batch_mean * (1 - decay))
#                 train_var = tf.assign(pop_var,
#                                       pop_var * decay + batch_var * (1 - decay))
#                 with tf.control_dependencies([train_mean, train_var]):
#                     self.outputs = act(tf.nn.batch_normalization(self.inputs,
#                         batch_mean, batch_var, beta, gamma, epsilon))
#             else:
#                 self.outputs = act(tf.nn.batch_normalization(self.inputs,
#                     pop_mean, pop_var, beta, gamma, epsilon))
#                     # self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, variance, beta, gamma, epsilon) )
#             # variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)  # 8 params in TF12 if zero_debias=True
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )

# class BatchNormLayer3(Layer):   # don't work http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     act : activation function.
#     is_train : boolean
#         Whether train or inference.
#     beta_init : beta initializer
#         The initializer for initializing beta
#     gamma_init : gamma initializer
#         The initializer for initializing gamma
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
#     - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         decay = 0.999,
#         epsilon = 0.00001,
#         act = tf.identity,
#         is_train = None,
#         beta_init = tf.zeros_initializer,
#         # gamma_init = tf.ones_initializer,
#         gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002),
#         name ='batchnorm_layer',
#     ):
#         """
#         Batch normalization on convolutional maps.
#         Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#         Args:
#             x:           Tensor, 4D BHWD input maps
#             n_out:       integer, depth of input maps
#             phase_train: boolean tf.Varialbe, true indicates training phase
#             scope:       string, variable scope
#         Return:
#             normed:      batch-normalized maps
#         """
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, act: %s, is_train: %s" %
#                             (self.name, decay, epsilon, act.__name__, is_train))
#         x_shape = self.inputs.get_shape()
#         params_shape = x_shape[-1:]
#
#         if is_train:
#             phase_train = tf.cast(tf.ones([]), tf.bool)
#         else:
#             phase_train = tf.cast(tf.zeros([]), tf.bool)
#
#         with tf.variable_scope(name) as vs:
#             gamma = tf.get_variable("gamma", shape=params_shape,
#                         initializer=gamma_init)
#             beta = tf.get_variable("beta", shape=params_shape,
#                         initializer=beta_init)
#             batch_mean, batch_var = tf.nn.moments(self.inputs, list(range(len(x_shape) - 1)),#[0,1,2],
#                             name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([batch_mean, batch_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)
#
#             mean, var = tf.cond(phase_train,
#                                 mean_var_with_update,
#                                 lambda: (ema.average(batch_mean), ema.average(batch_var)))
#             normed = tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon)
#             self.outputs = act( normed )
#             variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)    # 2 params beta, gamma
#                 # variables = [beta, gamma, moving_mean, moving_variance]
#
#             # print(len(variables))
#             # for idx, v in enumerate(variables):
#             #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
#             # exit()
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( variables )

# class BatchNormLayer_old(Layer):  # don't work
#     """
#     The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization``.
#
#     Batch normalization on fully-connected or convolutional maps.
#
#     Parameters
#     -----------
#     layer : a :class:`Layer` instance
#         The `Layer` class feeding into this layer.
#     decay : float
#         A decay factor for ExponentialMovingAverage.
#     epsilon : float
#         A small float number to avoid dividing by 0.
#     is_train : boolean
#         Whether train or inference.
#     name : a string or None
#         An optional name to attach to this layer.
#
#     References
#     ----------
#     - `tf.nn.batch_normalization <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.batch_normalization.md>`_
#     - `stackoverflow <http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow>`_
#     - `tensorflow.contrib <https://github.com/tensorflow/tensorflow/blob/b826b79718e3e93148c3545e7aa3f90891744cc0/tensorflow/contrib/layers/python/layers/layers.py#L100>`_
#     """
#     def __init__(
#         self,
#         layer = None,
#         act = tf.identity,
#         decay = 0.999,
#         epsilon = 0.001,
#         is_train = None,
#         name ='batchnorm_layer',
#     ):
#         Layer.__init__(self, name=name)
#         self.inputs = layer.outputs
#         print("  [TL] BatchNormLayer %s: decay: %f, epsilon: %f, is_train: %s" %
#                             (self.name, decay, epsilon, is_train))
#         if is_train == None:
#             raise Exception("is_train must be True or False")
#
#         # (name, input_var, decay, epsilon, is_train)
#         inputs_shape = self.inputs.get_shape()
#         axis = list(range(len(inputs_shape) - 1))
#         params_shape = inputs_shape[-1:]
#
#         with tf.variable_scope(name) as vs:
#             beta = tf.get_variable(name='beta', shape=params_shape,
#                                  initializer=tf.constant_initializer(0.0))
#             gamma = tf.get_variable(name='gamma', shape=params_shape,
#                                   initializer=tf.constant_initializer(1.0))
#             batch_mean, batch_var = tf.nn.moments(self.inputs,
#                                                 axis,
#                                                 name='moments')
#             ema = tf.train.ExponentialMovingAverage(decay=decay)
#
#             def mean_var_with_update():
#               ema_apply_op = ema.apply([batch_mean, batch_var])
#               with tf.control_dependencies([ema_apply_op]):
#                   return tf.identity(batch_mean), tf.identity(batch_var)
#
#             if is_train:
#                 is_train = tf.cast(tf.ones(1), tf.bool)
#             else:
#                 is_train = tf.cast(tf.zeros(1), tf.bool)
#
#             is_train = tf.reshape(is_train, [])
#
#             # print(is_train)
#             # exit()
#
#             mean, var = tf.cond(
#               is_train,
#               mean_var_with_update,
#               lambda: (ema.average(batch_mean), ema.average(batch_var))
#             )
#             normed = tf.nn.batch_normalization(
#               x=self.inputs,
#               mean=mean,
#               variance=var,
#               offset=beta,
#               scale=gamma,
#               variance_epsilon=epsilon,
#               name='tf_bn'
#             )
#         self.outputs = act( normed )
#
#         self.all_layers = list(layer.all_layers)
#         self.all_params = list(layer.all_params)
#         self.all_drop = dict(layer.all_drop)
#         self.all_layers.extend( [self.outputs] )
#         self.all_params.extend( [beta, gamma] )


class InstanceNormLayer(Layer):
    """The :class:`InstanceNormLayer` class is a for instance normalization.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function.
    epsilon : float
        A small float number.
    scale_init : beta initializer
        The initializer for initializing beta
    offset_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.
    """

    def __init__(
            self,
            layer=None,
            act=tf.identity,
            epsilon=1e-5,
            scale_init=tf.truncated_normal_initializer(mean=1.0, stddev=0.02),
            offset_init=tf.constant_initializer(0.0),
            name='instan_norm',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] InstanceNormLayer %s: epsilon:%f act:%s" % (self.name, epsilon, act.__name__))

        with tf.variable_scope(name) as vs:
            mean, var = tf.nn.moments(self.inputs, [1, 2], keep_dims=True)
            scale = tf.get_variable('scale', [self.inputs.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02), dtype=D_TYPE)
            offset = tf.get_variable('offset', [self.inputs.get_shape()[-1]], initializer=tf.constant_initializer(0.0), dtype=D_TYPE)
            self.outputs = scale * tf.div(self.inputs - mean, tf.sqrt(var + epsilon)) + offset
            self.outputs = act(self.outputs)
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


class LayerNormLayer(Layer):
    """
    The :class:`LayerNormLayer` class is for layer normalization, see `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    others : see  `tf.contrib.layers.layer_norm <https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm>`_
    """

    def __init__(self,
                 layer=None,
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

        if tf.__version__ < "1.3":
            raise Exception("Please use TF 1.3+")

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] LayerNormLayer %s: act:%s" % (self.name, act.__name__))
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

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)
