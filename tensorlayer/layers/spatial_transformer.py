#! /usr/bin/python
# -*- coding: utf-8 -*-

from six.moves import xrange

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import array_ops

from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorlayer.layers.core import TF_GRAPHKEYS_VARIABLES

from tensorlayer.layers.utils import flatten_reshape

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'transformer',
    'batch_transformer',
    'SpatialTransformer2dAffineLayer',
]


def transformer(U, theta, out_size, name='SpatialTransformer2dAffine'):
    """Spatial Transformer Layer for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__
    , see :class:`SpatialTransformer2dAffineLayer` class.

    Parameters
    ----------
    U : list of float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the localisation network should be [num_batch, 6], value range should be [0, 1] (via tanh).
    out_size: tuple of int
        The size of the output of the network (height, width)
    name: str
        Optional function name

    Returns
    -------
    Tensor
        The transformed tensor.

    References
    ----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    Notes
    -----
    To initialize the network to the identity transform init.

    >>> import tensorflow as tf
    >>> # ``theta`` to
    >>> identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    >>> identity = identity.flatten()
    >>> theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(
                tf.ones(shape=tf.stack([height, 1])),
                tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0])
            )
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

            output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer2dAffine'):
    """Batch Spatial Transformer function for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__.

    Parameters
    ----------
    U : list of float
        tensor of inputs [batch, height, width, num_channels]
    thetas : list of float
        a set of transformations for each input [batch, num_transforms, 6]
    out_size : list of int
        the size of the output [out_height, out_width]
    name : str
        optional function name

    Returns
    ------
    float
        Tensor of size [batch * num_transforms, out_height, out_width, num_channels]

    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i] * num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)


class SpatialTransformer2dAffineLayer(Layer):
    """The :class:`SpatialTransformer2dAffineLayer` class is a 2D `Spatial Transformer Layer <https://arxiv.org/abs/1506.02025>`__ for
    `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__.

    Parameters
    -----------
    prev_layer : :class:`Layer`
        Previous layer.
    theta_layer : :class:`Layer`
        The localisation network.
        - We will use a :class:`DenseLayer` to make the theta size to [batch, 6], value range to [0, 1] (via tanh).
    out_size : tuple of int or None
        The size of the output of the network (height, width), the feature maps will be resized by this.
    name : str
        A unique layer name.

    References
    -----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            theta_layer,
            out_size=None,
            name='spatial_trans_2d_affine',
    ):

        super(SpatialTransformer2dAffineLayer, self).__init__(prev_layer=[prev_layer, theta_layer], name=name)

        self.inputs = prev_layer.outputs  # Do not remove
        self.theta_layer = theta_layer

        if out_size is None:
            out_size = [40, 40]

        logging.info(
            "SpatialTransformer2dAffineLayer %s: in_size: %s out_size: %s" %
            (self.name, self.inputs.get_shape().as_list(), out_size)
        )

        with tf.variable_scope(name) as vs:

            # 1. make the localisation network to [batch, 6] via Flatten and Dense.
            if self.theta_layer.outputs.get_shape().ndims > 2:
                self.theta_layer.outputs = flatten_reshape(self.theta_layer.outputs, 'flatten')

            # 2. To initialize the network to the identity transform init.
            # 2.1 W
            n_in = int(self.theta_layer.outputs.get_shape()[-1])
            shape = (n_in, 6)

            W = tf.get_variable(name='W', initializer=tf.zeros(shape), dtype=LayersConfig.tf_dtype)
            # 2.2 b

            identity = tf.constant(np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten())

            b = tf.get_variable(name='b', initializer=identity, dtype=LayersConfig.tf_dtype)
            # 2.3 transformation matrix

            self.theta = tf.nn.tanh(tf.matmul(self.theta_layer.outputs, W) + b)
            # 3. Spatial Transformer Sampling
            # 3.1 transformation

            self.outputs = transformer(self.inputs, self.theta, out_size=out_size)

            # 3.2 automatically set batch_size and channels
            # e.g. [?, 40, 40, ?] --> [64, 40, 40, 1] or [64, 20, 20, 4]/ Hao Dong
            #
            fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value

            else:
                batch_size = array_ops.shape(self.inputs)[0]

            n_channels = self.inputs.get_shape().as_list()[-1]
            # logging.info(self.outputs)
            self.outputs = tf.reshape(self.outputs, shape=[batch_size, out_size[0], out_size[1], n_channels])
            # logging.info(self.outputs)
            # exit()
            # 4. Get all parameters
            variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

        # # theta_layer
        # self._add_layers(theta_layer.all_layers)
        # self._add_params(theta_layer.all_params)
        # self.all_drop.update(theta_layer.all_drop)

        self._add_layers(self.outputs)
        self._add_params(variables)
