#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.layers.core import Module

__all__ = [
    'DeformableConv2d',
]


class DeformableConv2d(Module):
    """The :class:`DeformableConv2d` class is a 2D
    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`__.

    Parameters
    ----------
    offset_layer : tl.Tensor
        To predict the offset of convolution operations.
        The shape is (batchsize, input height, input width, 2*(number of element in the convolution kernel))
        e.g. if apply a 3*3 kernel, the number of the last dimension should be 18 (2*3*3)
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels : int
        The number of in channels.
    name : str
        A unique layer name.

    Examples
    --------
    With TensorLayer
    >>> net = tl.layers.Input([5, 10, 10, 16], name='input')
    >>> offset1 = tl.layers.Conv2d(
    ...     n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME', name='offset1'
    ... )(net)
    >>> deformconv1 = tl.layers.DeformableConv2d(
    ...     offset_layer=offset1, n_filter=32, filter_size=(3, 3), name='deformable1'
    ... )(net)
    >>> offset2 = tl.layers.Conv2d(
    ...     n_filter=18, filter_size=(3, 3), strides=(1, 1), padding='SAME', name='offset2'
    ... )(deformconv1)
    >>> deformconv2 = tl.layers.DeformableConv2d(
    ...     offset_layer=offset2, n_filter=64, filter_size=(3, 3), name='deformable2'
    ... )(deformconv1)

    References
    ----------
    - The deformation operation was adapted from the implementation in `here <https://github.com/kastnerkyle/deform-conv>`__
    Notes
    -----
    - The padding is fixed to 'SAME'.
    - The current implementation is not optimized for memory usgae. Please use it carefully.

    """

    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
        self,
        offset_layer=None,
        # shape=(3, 3, 1, 100),
        n_filter=32,
        filter_size=(3, 3),
        act=None,
        padding='SAME',
        W_init=tl.initializers.truncated_normal(stddev=0.02),
        b_init=tl.initializers.constant(value=0.0),
        in_channels=None,
        name=None  # 'deformable_conv_2d',
    ):
        super().__init__(name, act=act)

        self.offset_layer = offset_layer
        self.n_filter = n_filter
        self.filter_size = filter_size
        self.padding = padding
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        self.kernel_n = filter_size[0] * filter_size[1]
        if self.offset_layer.get_shape()[-1] != 2 * self.kernel_n:
            raise AssertionError("offset.get_shape()[-1] is not equal to: %d" % 2 * self.kernel_n)

        logging.info(
            "DeformableConv2d %s: n_filter: %d, filter_size: %s act: %s" % (
                self.name, self.n_filter, str(self.filter_size
                                             ), self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def __repr__(self):
        actstr = self.act.__class__.__name__ if self.act is not None else 'No Activation'
        s = (
            '{classname}(in_channels={in_channels}, out_channels={n_filter}, kernel_size={filter_size}'
            ', padding={padding}'
        )
        if self.b_init is None:
            s += ', bias=False'
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):

        self.in_channels = inputs_shape[-1]

        self.input_h = int(inputs_shape[1])
        self.input_w = int(inputs_shape[2])
        initial_offsets = tl.ops.stack(
            tl.ops.meshgrid(tl.ops.range(self.filter_size[0]), tl.ops.range(self.filter_size[1]), indexing='ij')
        )  # initial_offsets --> (kh, kw, 2)
        initial_offsets = tl.ops.reshape(initial_offsets, (-1, 2))  # initial_offsets --> (n, 2)
        initial_offsets = tl.ops.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, n, 2)
        initial_offsets = tl.ops.expand_dims(initial_offsets, 0)  # initial_offsets --> (1, 1, n, 2)
        initial_offsets = tl.ops.tile(
            initial_offsets, [self.input_h, self.input_w, 1, 1]
        )  # initial_offsets --> (h, w, n, 2)
        initial_offsets = tl.ops.cast(initial_offsets, 'float32')
        grid = tl.ops.meshgrid(
            tl.ops.range(
                -int((self.filter_size[0] - 1) / 2.0), int(self.input_h - int((self.filter_size[0] - 1) / 2.0)), 1
            ),
            tl.ops.range(
                -int((self.filter_size[1] - 1) / 2.0), int(self.input_w - int((self.filter_size[1] - 1) / 2.0)), 1
            ), indexing='ij'
        )

        grid = tl.ops.stack(grid, axis=-1)
        grid = tl.ops.cast(grid, 'float32')  # grid --> (h, w, 2)
        grid = tl.ops.expand_dims(grid, 2)  # grid --> (h, w, 1, 2)
        grid = tl.ops.tile(grid, [1, 1, self.kernel_n, 1])  # grid --> (h, w, n, 2)
        self.grid_offset = grid + initial_offsets  # grid_offset --> (h, w, n, 2)

        self.filter_shape = (1, 1, self.kernel_n, self.in_channels, self.n_filter)

        self.W = self._get_weights("W_deformableconv2d", shape=self.filter_shape, init=self.W_init)

        if self.b_init:
            self.b = self._get_weights("b_deformableconv2d", shape=(self.n_filter, ), init=self.b_init)

        self.conv3d = tl.ops.Conv3D(strides=[1, 1, 1, 1, 1], padding='VALID')
        self.bias_add = tl.ops.BiasAdd()

    def forward(self, inputs):
        if self._forward_state == False:
            if self._built == False:
                self.build(tl.get_tensor_shape(inputs))
                self._built = True
            self._forward_state = True

        # shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
        offset = self.offset_layer
        grid_offset = self.grid_offset

        input_deform = self._tf_batch_map_offsets(inputs, offset, grid_offset)
        outputs = self.conv3d(input=input_deform, filters=self.W)
        outputs = tl.ops.reshape(
            tensor=outputs, shape=[outputs.get_shape()[0], self.input_h, self.input_w, self.n_filter]
        )
        if self.b_init:
            outputs = self.bias_add(outputs, self.b)
        if self.act:
            outputs = self.act(outputs)
        return outputs

    def _to_bc_h_w(self, x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tl.ops.transpose(a=x, perm=[0, 3, 1, 2])
        x = tl.ops.reshape(x, (-1, x_shape[1], x_shape[2]))
        return x

    def _to_b_h_w_n_c(self, x, x_shape):
        """(b*c, h, w, n) -> (b, h, w, n, c)"""
        x = tl.ops.reshape(x, (-1, x_shape[4], x_shape[1], x_shape[2], x_shape[3]))
        x = tl.ops.transpose(a=x, perm=[0, 2, 3, 4, 1])
        return x

    def tf_flatten(self, a):
        """Flatten tensor"""
        return tl.ops.reshape(a, [-1])

    def _get_vals_by_coords(self, inputs, coords, idx, out_shape):
        indices = tl.ops.stack(
            [idx, self.tf_flatten(coords[:, :, :, :, 0]),
             self.tf_flatten(coords[:, :, :, :, 1])], axis=-1
        )
        vals = tl.ops.gather_nd(inputs, indices)
        vals = tl.ops.reshape(vals, out_shape)
        return vals

    def _tf_repeat(self, a, repeats):
        """Tensorflow version of np.repeat for 1D"""
        # https://github.com/tensorflow/tensorflow/issues/8521

        if len(a.get_shape()) != 1:
            raise AssertionError("This is not a 1D Tensor")

        a = tl.ops.expand_dims(a, -1)
        a = tl.ops.tile(a, [1, repeats])
        a = self.tf_flatten(a)
        return a

    def _tf_batch_map_coordinates(self, inputs, coords):
        """Batch version of tf_map_coordinates
        Only supports 2D feature maps
        Parameters
        ----------
        inputs : ``tl.Tensor``
            shape = (b*c, h, w)
        coords : ``tl.Tensor``
            shape = (b*c, h, w, n, 2)
        Returns
        -------
        ``tl.Tensor``
            A Tensor with the shape as (b*c, h, w, n)
        """
        inputs_shape = inputs.get_shape()
        coords_shape = coords.get_shape()
        batch_channel = tl.get_tensor_shape(inputs)[0]
        input_h = int(inputs_shape[1])
        input_w = int(inputs_shape[2])
        kernel_n = int(coords_shape[3])
        n_coords = input_h * input_w * kernel_n

        coords_lt = tl.ops.cast(tl.ops.Floor()(coords), 'int32')
        coords_rb = tl.ops.cast(tl.ops.Ceil()(coords), 'int32')
        coords_lb = tl.ops.stack([coords_lt[:, :, :, :, 0], coords_rb[:, :, :, :, 1]], axis=-1)
        coords_rt = tl.ops.stack([coords_rb[:, :, :, :, 0], coords_lt[:, :, :, :, 1]], axis=-1)

        idx = self._tf_repeat(tl.ops.range(batch_channel), n_coords)

        vals_lt = self._get_vals_by_coords(inputs, coords_lt, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_rb = self._get_vals_by_coords(inputs, coords_rb, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_lb = self._get_vals_by_coords(inputs, coords_lb, idx, (batch_channel, input_h, input_w, kernel_n))
        vals_rt = self._get_vals_by_coords(inputs, coords_rt, idx, (batch_channel, input_h, input_w, kernel_n))

        coords_offset_lt = coords - tl.ops.cast(coords_lt, 'float32')

        vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, :, :, :, 0]
        vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, :, :, :, 0]
        mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, :, :, :, 1]

        return mapped_vals

    def _tf_batch_map_offsets(self, inputs, offsets, grid_offset):
        """Batch map offsets into input
        Parameters
        ------------
        inputs : ``tl.Tensor``
            shape = (b, h, w, c)
        offsets: ``tl.Tensor``
            shape = (b, h, w, 2*n)
        grid_offset: `tl.Tensor``
            Offset grids shape = (h, w, n, 2)
        Returns
        -------
        ``tl.Tensor``
            A Tensor with the shape as (b, h, w, c)
        """
        inputs_shape = inputs.get_shape()
        batch_size = tl.get_tensor_shape(inputs)[0]
        kernel_n = int(int(offsets.get_shape()[3]) / 2)
        input_h = inputs_shape[1]
        input_w = inputs_shape[2]
        channel = inputs_shape[3]

        # inputs (b, h, w, c) --> (b*c, h, w)
        inputs = self._to_bc_h_w(inputs, inputs_shape)

        # offsets (b, h, w, 2*n) --> (b, h, w, n, 2)
        offsets = tl.ops.reshape(offsets, (batch_size, input_h, input_w, kernel_n, 2))

        coords = tl.ops.expand_dims(grid_offset, 0)  # grid_offset --> (1, h, w, n, 2)
        coords = tl.ops.tile(coords, [batch_size, 1, 1, 1, 1]) + offsets  # grid_offset --> (b, h, w, n, 2)

        # clip out of bound
        coords = tl.ops.stack(
            [
                tl.ops.clip_by_value(coords[:, :, :, :, 0], 0.0, tl.ops.cast(input_h - 1, 'float32')),
                tl.ops.clip_by_value(coords[:, :, :, :, 1], 0.0, tl.ops.cast(input_w - 1, 'float32'))
            ], axis=-1
        )
        coords = tl.ops.tile(coords, [channel, 1, 1, 1, 1])

        mapped_vals = self._tf_batch_map_coordinates(inputs, coords)
        # (b*c, h, w, n) --> (b, h, w, n, c)
        mapped_vals = self._to_b_h_w_n_c(mapped_vals, [batch_size, input_h, input_w, kernel_n, channel])

        return mapped_vals
