#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorlayer.layers.core import Layer

from tensorlayer import tl_logging as logging

from tensorlayer.decorators import deprecated_alias

__all__ = [
    'UpSampling2dLayer',
    'DownSampling2dLayer',
]


class UpSampling2dLayer(Layer):
    """The :class:`UpSampling2dLayer` class is a up-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/api_docs/python/tf/image/resize_images>`__.

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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='upsample2d_layer',
    ):
        super(UpSampling2dLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "UpSampling2dLayer %s: is_scale: %s size: %s method: %d align_corners: %s" %
            (self.name, is_scale, size, method, align_corners)
        )

        if not isinstance(size, (list, tuple)) and len(size) == 2:
            raise AssertionError()

        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [int(size_h), int(size_w)]

        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [int(size_h), int(size_w)]

        else:
            raise Exception("Donot support shape %s" % self.inputs.get_shape())

        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(
                    self.inputs, size=size, method=method, align_corners=align_corners
                )
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(
                    self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners
                )

        self._add_layers(self.outputs)


class DownSampling2dLayer(Layer):
    """The :class:`DownSampling2dLayer` class is down-sampling 2D layer, see `tf.image.resize_images <https://www.tensorflow.org/versions/master/api_docs/python/image/resizing#resize_images>`__.

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

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            size,
            is_scale=True,
            method=0,
            align_corners=False,
            name='downsample2d_layer',
    ):
        super(DownSampling2dLayer, self).__init__(prev_layer=prev_layer, name=name)

        logging.info(
            "DownSampling2dLayer %s: is_scale: %s size: %s method: %d, align_corners: %s" %
            (self.name, is_scale, size, method, align_corners)
        )

        if not isinstance(size, (list, tuple)) and len(size) == 2:
            raise AssertionError()

        if len(self.inputs.get_shape()) == 3:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[0])
                size_w = size[1] * int(self.inputs.get_shape()[1])
                size = [int(size_h), int(size_w)]

        elif len(self.inputs.get_shape()) == 4:
            if is_scale:
                size_h = size[0] * int(self.inputs.get_shape()[1])
                size_w = size[1] * int(self.inputs.get_shape()[2])
                size = [int(size_h), int(size_w)]

        else:
            raise Exception("Do not support shape %s" % self.inputs.get_shape())

        with tf.variable_scope(name):
            try:
                self.outputs = tf.image.resize_images(
                    self.inputs, size=size, method=method, align_corners=align_corners
                )
            except Exception:  # for TF 0.10
                self.outputs = tf.image.resize_images(
                    self.inputs, new_height=size[0], new_width=size[1], method=method, align_corners=align_corners
                )

        self._add_layers(self.outputs)
