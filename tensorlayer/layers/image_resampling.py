#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer.decorators import auto_parse_inputs
from tensorlayer.decorators import deprecated_alias
from tensorlayer.decorators import deprecated_args

__all__ = [
    'UpSampling2dLayer',
    'DownSampling2dLayer',
]


class UpSampling2dLayer(Layer):
    """The :class:`UpSampling2dLayer` class is a up-sampling 2D layer, see `tf.image.resize_images
    <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with 4-D Tensor of the shape (batch, height, width, channels) or 3-D Tensor of the shape (height, width, channels).
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is a scale factor; otherwise, the `size` is the numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, align the corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        size=list(),
        is_scale=True,
        method=0,
        align_corners=False,
        name='upsample2d_layer',
    ):

        if not isinstance(size, (list, tuple)):
            raise AssertionError("`size` argument should be a `list` or a `tuple`")

        if len(size) != 2:
            raise AssertionError("`size` argument should be of length 2")

        self.prev_layer = prev_layer
        self.size = size
        self.is_scale = is_scale
        self.method = method
        self.align_corners = align_corners
        self.name = name

        super(UpSampling2dLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("is_scale: %s" % self.is_scale)
        except AttributeError:
            pass

        try:
            additional_str.append("size: {}".format(self.size))
        except AttributeError:
            pass

        try:
            additional_str.append("method: %s" % self.method)
        except AttributeError:
            pass

        try:
            additional_str.append("align_corners: %s" % self.align_corners)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        if len(self.inputs.shape) == 3:
            x_pos, y_pos = (0, 1)

        elif len(self.inputs.shape) == 4:
            x_pos, y_pos = (1, 2)

        else:
            raise RuntimeError("The input shape: %s is not supported" % tf.shape(self.inputs))

        with tf.variable_scope(self.name):

            if self.is_scale:
                if all(isinstance(x, int) for x in self.size):
                    if None not in [self.inputs.get_shape()[x_pos]._value, self.inputs.get_shape()[y_pos]._value]:
                        size_h = self.inputs.get_shape()[x_pos] * self.size[0]
                        size_w = self.inputs.get_shape()[y_pos] * self.size[1]
                    else:
                        size_h = tf.shape(self.inputs)[x_pos] * self.size[0]
                        size_w = tf.shape(self.inputs)[y_pos] * self.size[1]

                    _size = [size_h, size_w]

                else:
                    raise ValueError("all elements of tuple `size` hyperparameter should of type `int`")

            else:
                _size = self.size

            print("size:", _size)

            self.outputs = tf.image.resize_images(
                self.inputs, size=_size, method=self.method, align_corners=self.align_corners
            )
            self.outputs = tf.cast(self.outputs, self.inputs.dtype)

            self.out_shape = self.outputs.get_shape()

        self._add_layers(self.outputs)


class DownSampling2dLayer(Layer):
    """The :class:`DownSampling2dLayer` class is down-sampling 2D layer, see `tf.image.resize_images
    <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer with 4-D Tensor in the shape of (batch, height, width, channels) or 3-D Tensor in the shape of (height, width, channels).
    size : tuple of int/float
        (height, width) scale factor or new size of height and width.
    is_scale : boolean
        If True (default), the `size` is the scale factor; otherwise, the `size` are numbers of pixels of height and width.
    method : int
        The resize method selected through the index. Defaults index is 0 which is ResizeMethod.BILINEAR.
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
    align_corners : boolean
        If True, exactly align all 4 corners of the input and output. Default is False.
    name : str
        A unique layer name.

    """

    @deprecated_alias(
        layer='prev_layer', end_support_version="2.0.0"
    )  # TODO: remove this line before releasing TL 2.0.0
    @deprecated_args(
        end_support_version="2.1.0",
        instructions="`prev_layer` is deprecated, use the functional API instead",
        deprecated_args=("prev_layer", ),
    )  # TODO: remove this line before releasing TL 2.1.0
    def __init__(
        self,
        prev_layer=None,
        size=list(),
        is_scale=True,
        method=0,
        align_corners=False,
        name='downsample2d_layer',
    ):

        if not isinstance(size, (list, tuple)):
            raise AssertionError("`size` argument should be a `list` or a `tuple`")

        if len(size) != 2:
            raise AssertionError("`size` argument should be of length 2")

        self.prev_layer = prev_layer
        self.size = size
        self.is_scale = is_scale
        self.method = method
        self.align_corners = align_corners
        self.name = name

        super(DownSampling2dLayer, self).__init__()

    def __str__(self):
        additional_str = []

        try:
            additional_str.append("is_scale: %s" % self.is_scale)
        except AttributeError:
            pass

        try:
            additional_str.append("size: {}".format(self.size))
        except AttributeError:
            pass

        try:
            additional_str.append("method: %s" % self.method)
        except AttributeError:
            pass

        try:
            additional_str.append("align_corners: %s" % self.align_corners)
        except AttributeError:
            pass

        try:
            additional_str.append("out_shape: %s" % self.out_shape)
        except AttributeError:
            pass

        return self._str(additional_str)

    @auto_parse_inputs
    def compile(self, prev_layer, is_train=True):

        if len(self.inputs.shape) == 3:
            x_pos, y_pos = (0, 1)

        elif len(self.inputs.shape) == 4:
            x_pos, y_pos = (1, 2)

        else:
            raise RuntimeError("The input shape: %s is not supported" % tf.shape(self.inputs))

        with tf.variable_scope(self.name):

            if self.is_scale:
                if all(isinstance(x, int) for x in self.size):
                    if None not in [self.inputs.get_shape()[x_pos]._value, self.inputs.get_shape()[y_pos]._value]:
                        size_h = tf.cast(tf.ceil(int(self.inputs.get_shape()[x_pos]) / float(self.size[0])), tf.int32)
                        size_w = tf.cast(tf.ceil(int(self.inputs.get_shape()[y_pos]) / float(self.size[1])), tf.int32)
                    else:
                        size_h = tf.cast(tf.ceil(tf.shape(self.inputs)[x_pos] / np.float32(self.size[0])), tf.int32)
                        size_w = tf.cast(tf.ceil(tf.shape(self.inputs)[y_pos] / np.float32(self.size[1])), tf.int32)

                elif all(isinstance(x, float) for x in self.size):
                    if None not in [self.inputs.get_shape()[x_pos]._value, self.inputs.get_shape()[y_pos]._value]:
                        size_h = tf.cast(int(self.inputs.get_shape()[x_pos]) * self.size[0], tf.int32)
                        size_w = tf.cast(int(self.inputs.get_shape()[y_pos]) * self.size[1], tf.int32)
                    else:
                        size_h = tf.shape(self.inputs)[x_pos] * self.size[0]
                        size_w = tf.shape(self.inputs)[y_pos] * self.size[1]

                else:
                    raise ValueError("all elements of tuple `size` hyperparameter should be either `int` or `float`")

                _size = [size_h, size_w]

            else:
                _size = self.size

            self.outputs = tf.image.resize_images(
                self.inputs, size=_size, method=self.method, align_corners=self.align_corners
            )
            self.outputs = tf.cast(self.outputs, self.inputs.dtype)

            self.out_shape = self.outputs.get_shape()

        self._add_layers(self.outputs)
