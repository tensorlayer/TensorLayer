#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops

import tensorlayer as tl

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.layers.utils.reshape import flatten_reshape
from tensorlayer.layers.utils.spatial_transformer import transformer

from tensorlayer.decorators import private_method
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = ['SpatialTransformer2dAffineLayer']


class SpatialTransformer2dAffineLayer(Layer):
    """The :class:`SpatialTransformer2dAffineLayer` class is a 2D `Spatial Transformer Layer <https://arxiv.org/abs/1506.02025>`__ for
    `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__.

    Parameters
    -----------
    out_size : tuple of int or None
        The size of the output of the network (height, width), the feature maps will be resized by this.
    name : str
        A unique layer name.

    References
    -----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    """

    def __init__(
        self,
        out_size,
        name='spatial_trans_2d_affine',
    ):

        if out_size is None:
            out_size = [40, 40]

        self.out_size = out_size
        self.name = name

        super(SpatialTransformer2dAffineLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("in_size: %s" % self._temp_data['inputs'][0].outputs.shape)
        except AttributeError:
            pass

        try:
            additional_str.append("out_size: %s" % self.out_size)
        except AttributeError:
            pass

        return self._str(additional_str)

    def __call__(self, prev_layer, theta_layer, is_train=True):
        """
        prev_layer : :class:`Layer`
            Previous layer.
        theta_layer : :class:`Layer`
            The localisation network.
            - We will use a :class:`DenseLayer` to make the theta size to [batch, 6], value range to [0, 1] (via tanh).
        is_train: boolean (default: True)
            Set the TF Variable in training mode and may impact the behaviour of the layer.
        """
        return super(SpatialTransformer2dAffineLayer, self).__call__(
            prev_layer=[prev_layer, theta_layer], is_train=is_train
        )

    def build(self):

        input_layer = self._temp_data['inputs'][0]
        theta_layer = self._temp_data['inputs'][1]

        with tf.variable_scope(self.name) as vs:

            # 1. make the localisation network to [batch, 6] via Flatten and Dense.
            if theta_layer.outputs.get_shape().ndims > 2:
                theta_layer.outputs = flatten_reshape(theta_layer.outputs, 'flatten')

            # 2. To initialize the network to the identity transform init.
            # 2.1 W
            w_shape = (int(theta_layer.outputs.get_shape()[-1]), 6)

            weight_matrix = self._get_tf_variable(
                name='W',
                dtype=input_layer.dtype,
                trainable=self._temp_data['is_train'],
                initializer=tf.zeros(w_shape),
            )

            # 2.2 b
            identity = tf.constant(np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten())

            b = self._get_tf_variable(
                name='b',
                dtype=input_layer.dtype,
                trainable=self._temp_data['is_train'],
                initializer=identity,
            )

            # 2.3 transformation matrix
            self.theta = tf.nn.tanh(tf.matmul(theta_layer.outputs, weight_matrix) + b)

            # 3. Spatial Transformer Sampling
            # 3.1 transformation

            self._temp_data['outputs'] = transformer(input_layer, self.theta, out_size=self.out_size)

            # 3.2 automatically set batch_size and channels
            # e.g. [?, 40, 40, ?] --> [64, 40, 40, 1] or [64, 20, 20, 4]/ Hao Dong
            #
            fixed_batch_size = input_layer.get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value

            else:
                batch_size = array_ops.shape(input_layer)[0]

            n_channels = input_layer.get_shape().as_list()[-1]

            self._temp_data['outputs'] = tf.reshape(
                self._temp_data['outputs'], shape=[batch_size, self.out_size[0], self.out_size[1], n_channels]
            )

            # 4. Get all parameters
            self._temp_data['local_weights'] = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)
