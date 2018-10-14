#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import logging

from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args
from tensorlayer.decorators import private_method

__all__ = [
    'DeformableConv2d',
]


class DeformableConv2d(Layer):
    """The :class:`DeformableConv2d` class is a 2D
    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`__.

    Parameters
    ----------
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : str
        A unique layer name.

    Examples
    --------
    >>> net = tl.layers.Input(x, name='input_layer')
    >>> offset1 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=act, padding='SAME', name='offset1')
    >>> net = tl.layers.DeformableConv2d(net, offset1, 32, (3, 3), act=act, name='deformable1')
    >>> offset2 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=act, padding='SAME', name='offset2')
    >>> net = tl.layers.DeformableConv2d(net, offset2, 64, (3, 3), act=act, name='deformable2')

    References
    ----------
    - The deformation operation was adapted from the implementation in `here <https://github.com/felixlaumon/deform-conv>`__

    Notes
    -----
    - The padding is fixed to 'SAME'.
    - The current implementation is not optimized for memory usgae. Please use it carefully.

    """

    def __init__(
        self,
        n_filter=32,
        filter_size=(3, 3),
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args=None,
        b_init_args=None,
        act=None,
        name='deformable_conv_2d',
    ):

        self.n_filter = n_filter
        self.filter_size = filter_size
        self.W_init = W_init
        self.b_init = b_init
        self.act = act
        self.name = name

        super(DeformableConv2d, self).__init__(W_init_args=W_init_args, b_init_args=b_init_args)

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("n_filter: %d" % self.n_filter)
        except AttributeError:
            pass

        try:
            additional_str.append("filter_size: %s" % str(self.filter_size))
        except AttributeError:
            pass

        return self._str(additional_str)

    def __call__(self, prev_layer, offset_layer, is_train=True):
        """
        prev_layer : :class:`Layer`
            Previous layer.
        offset_layer : :class:`Layer`
            To predict the offset of convolution operations.
            The output shape is (batchsize, input height, input width, 2*(number of element in the convolution kernel))
            e.g. if apply a 3*3 kernel, the number of the last dimension should be 18 (2*3*3)
        is_train: boolean (default: True)
            Set the TF Variable in training mode and may impact the behaviour of the layer.
        """
        return super(DeformableConv2d, self).__call__(prev_layer=[prev_layer, offset_layer], is_train=is_train)

    def build(self):

        input_layer = self._temp_data['inputs'][0]
        offset_layer = self._temp_data['inputs'][1]

        input_dtype = input_layer.dtype

        try:
            input_channels = int(input_layer.get_shape()[-1])

        except TypeError:  # if input_channels is ?, it happens when using Spatial Transformer Net
            input_channels = 1
            logging.warning("unknown input channels, set to 1")

        w_shape = (self.filter_size[0], self.filter_size[1], input_channels, self.n_filter)

        with tf.variable_scope(self.name):

            if offset_layer.get_shape()[-1] != 2 * w_shape[0] * w_shape[1]:
                raise AssertionError("offset_layer.get_shape()[-1] is not equal to: %d" % (2 * w_shape[0] * w_shape[1]))

            # Grid initialisation
            input_h = int(input_layer.get_shape()[1])
            input_w = int(input_layer.get_shape()[2])

            kernel_n = w_shape[0] * w_shape[1]

            # initial_offsets --> (kh, kw, 2)
            initial_offsets = tf.stack(tf.meshgrid(tf.range(w_shape[0]), tf.range(w_shape[1]), indexing='ij'))

            # initial_offsets --> (n, 2)
            initial_offsets = tf.reshape(initial_offsets, (-1, 2))

            # initial_offsets --> (1, n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)

            # initial_offsets --> (1, 1, n, 2)
            initial_offsets = tf.expand_dims(initial_offsets, 0)

            # initial_offsets --> (h, w, n, 2)
            initial_offsets = tf.tile(initial_offsets, [input_h, input_w, 1, 1])

            initial_offsets = tf.cast(initial_offsets, input_dtype)

            grid = tf.meshgrid(
                tf.range(-int((w_shape[0] - 1) / 2.0), int(input_h - int((w_shape[0] - 1) / 2.0)), 1),
                tf.range(-int((w_shape[1] - 1) / 2.0), int(input_w - int((w_shape[1] - 1) / 2.0)), 1),
                indexing='ij'
            )

            grid = tf.stack(grid, axis=-1)

            # grid --> (h, w, 2)
            grid = tf.cast(grid, input_dtype)

            # grid --> (h, w, 1, 2)
            grid = tf.expand_dims(grid, 2)

            # grid --> (h, w, n, 2)
            grid = tf.tile(grid, [1, 1, kernel_n, 1])

            # grid_offset --> (h, w, n, 2)
            grid_offset = grid + initial_offsets

            input_deform = self._tf_batch_map_offsets(input_layer, offset_layer, grid_offset)

            weight_matrix = self._get_tf_variable(
                name='W_deformableconv2d',
                shape=(1, 1, w_shape[0] * w_shape[1], w_shape[-2], w_shape[-1]),
                dtype=input_dtype,
                trainable=self._temp_data['is_train'],
                initializer=self.W_init,
                **self.W_init_args
            )

            _tensor = tf.nn.conv3d(input_deform, weight_matrix, strides=[1, 1, 1, 1, 1], padding='VALID', name=None)

            if self.b_init:
                b = self._get_tf_variable(
                    name='b_deformableconv2d',
                    shape=(w_shape[-1]),
                    dtype=input_dtype,
                    trainable=self._temp_data['is_train'],
                    initializer=self.b_init,
                    **self.b_init_args
                )

                _tensor = tf.nn.bias_add(_tensor, b, name='bias_add')

            _tensor = self._apply_activation(_tensor)

            self._temp_data['outputs'] = tf.reshape(
                tensor=_tensor, shape=[tf.shape(input_layer)[0], input_h, input_w, w_shape[-1]]
            )

    @private_method
    def _to_bc_h_w(self, x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, x_shape[1], x_shape[2]))
        return x

    @private_method
    def _to_b_h_w_n_c(self, x, x_shape):
        """(b*c, h, w, n) -> (b, h, w, n, c)"""
        x = tf.reshape(x, (-1, x_shape[4], x_shape[1], x_shape[2], x_shape[3]))
        x = tf.transpose(x, [0, 2, 3, 4, 1])
        return x

    @private_method
    def tf_flatten(self, a):
        """Flatten tensor"""
        return tf.reshape(a, [-1])

    @private_method
    def _get_vals_by_coords(self, inputs, coords, idx, out_shape):
        indices = tf.stack(
            [idx, self.tf_flatten(coords[:, :, :, :, 0]),
             self.tf_flatten(coords[:, :, :, :, 1])], axis=-1
        )
        vals = tf.gather_nd(inputs, indices)
        vals = tf.reshape(vals, out_shape)
        return vals

    @private_method
    def _tf_repeat(self, a, repeats):
        """Tensorflow version of np.repeat for 1D"""
        # https://github.com/tensorflow/tensorflow/issues/8521

        if len(a.get_shape()) != 1:
            raise AssertionError("This is not a 1D Tensor")

        a = tf.expand_dims(a, -1)
        a = tf.tile(a, [1, repeats])
        a = self.tf_flatten(a)
        return a

    @private_method
    def _tf_batch_map_coordinates(self, inputs, coords):
        """Batch version of tf_map_coordinates

        Only supports 2D feature maps

        Parameters
        ----------
        inputs : ``tf.Tensor``
            shape = (b*c, h, w)
        coords : ``tf.Tensor``
            shape = (b*c, h, w, n, 2)

        Returns
        -------
        ``tf.Tensor``
            A Tensor with the shape as (b*c, h, w, n)

        """
        input_shape = inputs.get_shape()
        coords_shape = coords.get_shape()

        batch_channel = tf.shape(inputs)[0]

        input_dtype = inputs.dtype

        input_h = int(input_shape[1])
        input_w = int(input_shape[2])
        kernel_n = int(coords_shape[3])
        n_coords = input_h * input_w * kernel_n

        coords_lt = tf.cast(tf.floor(coords), 'int32')
        coords_rb = tf.cast(tf.ceil(coords), 'int32')
        coords_lb = tf.stack([coords_lt[:, :, :, :, 0], coords_rb[:, :, :, :, 1]], axis=-1)
        coords_rt = tf.stack([coords_rb[:, :, :, :, 0], coords_lt[:, :, :, :, 1]], axis=-1)

        idx = self._tf_repeat(tf.range(batch_channel), n_coords)

        vals_lt = self._get_vals_by_coords(inputs, coords_lt, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_rb = self._get_vals_by_coords(inputs, coords_rb, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_lb = self._get_vals_by_coords(inputs, coords_lb, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_rt = self._get_vals_by_coords(inputs, coords_rt, idx, (batch_channel, input_h, input_w, kernel_n))

        coords_offset_lt = coords - tf.cast(coords_lt, input_dtype)

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, :, :, 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, :, :, 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, :, :, 1]

        return mapped_vals

    @private_method
    def _tf_batch_map_offsets(self, inputs, offsets, grid_offset):
        """Batch map offsets into input

        Parameters
        ------------
        inputs : ``tf.Tensor``
            shape = (b, h, w, c)
        offsets: ``tf.Tensor``
            shape = (b, h, w, 2*n)
        grid_offset: `tf.Tensor``
            Offset grids shape = (h, w, n, 2)

        Returns
        -------
        ``tf.Tensor``
            A Tensor with the shape as (b, h, w, c)

        """
        input_shape = inputs.get_shape()
        input_dtype = inputs.dtype
        batch_size = tf.shape(inputs)[0]
        kernel_n = int(int(offsets.get_shape()[3]) / 2)
        input_h = input_shape[1]
        input_w = input_shape[2]
        channel = input_shape[3]

        # inputs (b, h, w, c) --> (b*c, h, w)
        inputs = self._to_bc_h_w(inputs, input_shape)

        # offsets (b, h, w, 2*n) --> (b, h, w, n, 2)
        offsets = tf.reshape(offsets, (batch_size, input_h, input_w, kernel_n, 2))
        # offsets (b, h, w, n, 2) --> (b*c, h, w, n, 2)
        # offsets = tf.tile(offsets, [channel, 1, 1, 1, 1])

        coords = tf.expand_dims(grid_offset, 0)  # grid_offset --> (1, h, w, n, 2)
        coords = tf.cast(tf.tile(coords, [batch_size, 1, 1, 1, 1]),
                         input_dtype) + offsets  # grid_offset --> (b, h, w, n, 2)

        # clip out of bound
        coords = tf.stack(
            [
                tf.clip_by_value(coords[:, :, :, :, 0], 0.0, tf.cast(input_h - 1, input_dtype)),
                tf.clip_by_value(coords[:, :, :, :, 1], 0.0, tf.cast(input_w - 1, input_dtype))
            ],
            axis=-1
        )
        coords = tf.tile(coords, [channel, 1, 1, 1, 1])

        mapped_vals = self._tf_batch_map_coordinates(inputs, coords)
        # (b*c, h, w, n) --> (b, h, w, n, c)
        mapped_vals = self._to_b_h_w_n_c(mapped_vals, [batch_size, input_h, input_w, kernel_n, channel])

        return mapped_vals
